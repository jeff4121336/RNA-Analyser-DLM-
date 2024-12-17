import glob, numpy as np, fm
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch
from Bio import SeqIO
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from dslayer import RNAClassifier, RNATypeDataset, EarlyStopper
import time, random, sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score

torch.autograd.set_detect_anomaly(True)

#hyperparameters
labels_ref = {"5S_rRNA": 0, "5_8S_rRNA": 1, "tRNA": 2, "ribozyme": 3, "CD-box": 4, "miRNA": 5,
		"Intron_gpI": 6, "Intron_gpII": 7,  "scaRNA": 8, "HACA-box": 9, "riboswitch": 10, "IRES": 11, "leader": 12, "mRNA": 13
    ,"unknown": 14, "pad": 15}
reverse_labels_ref = {v: k for k, v in labels_ref.items()}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
padding_token = '-'

# Training hyperparameters
batch_size = 16
lr = 1e-3
epochs = 100
intermediate_evel = 5 # involves early stopper mechanism

# Model hyperparameters 
num_channels = 30
kernel_size = 5
dropout_rate = 0.2
padding = 2

def calculate_binary_metrics(y_true, y_pred, target_class):
	# Convert labels to binary: 1 for the target class, 0 for all others

	binary_y_true = (y_true == target_class).astype(int)
	binary_y_pred = (y_pred == target_class).astype(int)
	
	# Calculate confusion matrix
	cm = confusion_matrix(binary_y_true, binary_y_pred)

	# Extract metrics from confusion matrix
	TP = cm[1, 1]  # True Positives
	TN = cm[0, 0]  # True Negatives
	FP = cm[0, 1]  # False Positives
	FN = cm[1, 0]  # False Negatives
	
	# Calculate precision, recall, and F1 score
	precision = precision_score(binary_y_true, binary_y_pred)
	recall = recall_score(binary_y_true, binary_y_pred)
	f1 = f1_score(binary_y_true, binary_y_pred)
	mcc = matthews_corrcoef(binary_y_true, binary_y_pred)
	acc = accuracy_score(binary_y_true, binary_y_pred)
	return {
		'Confusion Matrix': cm,
		'True Positives': TP,
		'True Negatives': TN,
		'False Positives': FP,
		'False Negatives': FN,
		'Accuracy': acc,
		'Precision': precision,
		'Recall': recall,
		'F1 Score': f1,
		'MCC': mcc
	}

def calculate_metric_with_sklearn(predictions, labels):
	valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
	valid_predictions = predictions[valid_mask]
	valid_labels = labels[valid_mask]

	return {
		"accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
		"precision": sklearn.metrics.precision_score(
				valid_labels, valid_predictions, average="macro", zero_division=0
		),
		"recall": sklearn.metrics.recall_score(
				valid_labels, valid_predictions, average="macro", zero_division=0
		),
		"f1": sklearn.metrics.f1_score(
				valid_labels, valid_predictions, average="macro", zero_division=0
		),
		"matthews_correlation": sklearn.metrics.matthews_corrcoef(
				valid_labels, valid_predictions
		),
	}

def crop_sequences(sequences_labels, segment_length=32, overlap=12):
	"""
		Crop input RNA into smaller parts for training and features mapping purpose.
		Allow list of sequences.

		return segments
	"""
	# example data in sequences_labels,
	# (('RF00168_ACCK01000036_1_8201-8035', 'UAUAAAGAUAGAUGUUGUCUUGGUGAUUCAGGACAUGAUGAAGAGGCGUUAUCUUAUCG'), 10)
	segments = [[] for _ in range(len(sequences_labels))]
	sequence = [seq[1] for seq, _ in sequences_labels]
	labels = [lb for _, lb in sequences_labels]

	for i, seq in enumerate (sequence):
		# print(f"Processing sequence {i}: {seq}")
		# seq = ''.join(maps[nucleotide] for nucleotide in seq)
		for j in range(0, len(seq), segment_length - overlap):
			if j + segment_length >= len(seq):
				remaining_segment = seq[j:len(seq)]
				padded_segment = remaining_segment + (padding_token * (segment_length-len(remaining_segment)))
				segments[i].append((padded_segment, labels[i])) 
				# print(f"Segment (_if_) added: {segments[i]}")
				break
			else:
				segments[i].append((seq[j :(j+segment_length)], labels[i]))
				# print(f"Segment (else) added: {segments[i]}")
		# time.sleep(1)
	return segments

def label_data(filenames):
	"""
		Iterate the fasta files, label all the data. Print out count of labels.

		return sequences_labels_pairs, 2D array with tuples for train/test/eval data.
	"""
	sequences_labels_pairs = [[] for i in range(0, 3)] # Train, Test, Eval
	
	for i, filename in enumerate (filenames):

		print(f"Processing Files: {filename}")
		datas = list(SeqIO.parse(filename, 'fasta'))
	
		for data in datas:

			sequence_desc = data.description.split()[-1]  # Get the definition line (sequence ID)
			sequence_id = data.id
			sequence = ''.join({'T' : 'U'}.get(base, base) for base in str(data.seq)) # map dna to rna
	
			seq = (sequence_id, sequence)
			label = labels_ref.get(sequence_desc, labels_ref["unknown"])
			
			sequences_labels_pairs[i].append((seq, label))
	
		label_counts = Counter(label for _, label in sequences_labels_pairs[i]) # count each label occurences 
		for label, count in label_counts.items():
			print(f"{reverse_labels_ref[label]}: {count} ")

	return sequences_labels_pairs 

def generate_embed(dataset, model, segment_length):
	"""
		A 2D array pass in, [ [(segment, label), (segment,label)], [(segment, label)...] ... ] and
		model for generating the embedding.

		Generate embedding for each segments in each sequences. Implement chunk size mechanism 
		Save token embeddings, labels for classification use.
		
		return embedding in segments for all sequences, len(sequence) of y labels
	"""	
	token_embeddings = []
	y_labels = []
	sequence_str = [] # this will save sequences in batches, it means this may be cut due to maximum length
	sequence_idx = [] # this will save the idx of sequences, this will be refer when we want to reconstruct the input data
	segment_batch_size = 30
	maximum_length = segment_batch_size * segment_length # 960
	# print(model)
	for i in tqdm(range(len(dataset))):
		# All segments for one sequence
		segments_pairs = dataset[i]
		segments = [seg for seg, _ in segments_pairs]
		labels = [label for _, label in segments_pairs]
		# print(f"f:{labels}")
		# Prepare segment tuples
		segment_tuple = []
		for k in range(0, len(segments), segment_batch_size):
			segment = ''.join(segments[k:k + segment_batch_size])
			sequence_id = f"Sequence {i} Segment {k} to {min(k + segment_batch_size, len(segments)) - 1}"
			segment_padding = '-' * (maximum_length - len(segment))
			segment = segment + segment_padding
			segment_tuple.append((sequence_id, segment))
		
		
		if len(labels) % segment_batch_size != 0:
			labels = labels + [15] * (segment_batch_size - (len(labels) % segment_batch_size))
	
		# emb will be (N, 34, 640) for each sequence
		# batch_token.shpe will be (N, 34)
		# Maximum length set to 962 (1024 in paper), segment_batch_size = (962 - 2) // 32 = 30, len(segment) pad to 1024 with empty token

		batch_labels, batch_strs, batch_tokens = batch_converter(segment_tuple)
		# print(len(batch_tokens[0]))
		
		with torch.no_grad():
			results = model(batch_tokens.to(device), repr_layers=[12])
			emb = results['representations'][12].cpu().numpy()  # Shape: (Num of segments, Length of each segments, embedding dimension)
		
		
		# for emb != (1, 964, 640):
		for j in range(emb.shape[0]):
			token_embeddings.append(emb[j:j+1])

		labels = np.array(labels)

		for j in range(labels.shape[0] // segment_batch_size):
			y_labels.append(labels[segment_batch_size * j : segment_batch_size * (j+1)]) 
			# print(f"{j} : {labels[segment_batch_size * j : segment_batch_size * (j+1)]}")
			# if labels.shape != (segment_batch_size, ):
			# 	# print(labels)
			# 	print(labels.shape)
			# print(labels[segment_batch_size * i: segment_batch_size * (i+1)].shape)

		for _, strs in enumerate(batch_strs):
			sequence_str.append(strs)
		
		sequence_idx.append((i, i + len(batch_strs) - 1))
			
	print(len(token_embeddings), len(y_labels), len(sequence_str))
	token_embeddings = np.concatenate(token_embeddings, axis=0)  
	token_embeddings = token_embeddings[:, 1:-1, :]
	# print(token_embeddings.shape)
	y_labels = np.array(y_labels) 
	sequence_str = np.array(sequence_str)
	sequence_idx = np.array(sequence_idx)

	return token_embeddings, y_labels, sequence_str, sequence_idx

def train_result(train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_epoch):
	plt.figure(figsize=(8, 6))
	plt.plot(train_loss_history, label='train loss')
	plt.plot(val_loss_history, label='val loss')

	# the epoch with best validation loss
	plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.8)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss History')
	plt.legend()

	plt.savefig("loss_history.png")

	plt.figure(figsize=(8, 6))
	plt.plot(train_acc_history, label='train accuracy')
	plt.plot(val_acc_history, label='val accuracy')

	# the epoch with best validation accuracy
	plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.8)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Accuracy History')
	plt.legend()
	
	plt.savefig("accuracy.png")

def train(x_train, y_train, x_val, y_val):

	train_dataset = RNATypeDataset(x_train, y_train)
	val_dataset = RNATypeDataset(x_val, y_val)
	
	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

	model = RNAClassifier(len(labels_ref), num_channels, kernel_size, dropout_rate, padding).to(device)
	# print(model)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	max_val_acc = -1
	best_epoch = -1

	train_loss_history = []
	val_loss_history = []

	train_acc_history = []
	val_acc_history = []

	early_stopper = EarlyStopper(patience=2, min_delta=0.01)

	for epoch in tqdm(range(epochs)):
		# train the model
		train_losses = []
		train_preds = []
		train_targets = []
	
		model.train()

		for batch in train_loader:
			x, y = batch
			x, y = x.to(device).float(), y.to(device).long()
			y_final = y[:, 0]  # Final label (shape: (B,))
			
			# Get model predictions
			y_pred, feature_maps = model(x)
			
			# Compute final output by summing over segments
			final_output = torch.sum(y_pred, dim=1)
			final_output[:, -2:] = 0  # Adjust as required
			
			# Calculate loss for the final label
			# total_loss = criterion(final_output, y_final)
			total_loss = criterion(y_pred.view(-1, y_pred.shape[-1]), y.view(-1)) 
			# total_loss = 0.8 * final_label_loss + 0.2 * segment_loss

		
			prob, y_pred_indices = torch.max(final_output, dim=1)

			# Store predictions and targets
			train_losses.append(total_loss.item())
			train_preds.append(y_pred_indices)
			train_targets.append(y_final)

			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

		# validate the model
		val_losses = []
		val_preds = []
		val_targets = []

		model.eval()

		for batch in val_loader:
			x, y = batch
			x, y = x.to(device).float(), y.to(device).long()
			y_final = y[:, 0]  # Final label (shape: (B,))
			
			# Get model predictions
			y_pred, feature_maps = model(x)
			
			# Compute final output by summing over segments
			final_output = torch.sum(y_pred, dim=1)
			final_output[:, -2:] = 0  # Adjust as required
			
			# Calculate loss for the final label
			# total_loss = criterion(final_output, y_final)
			total_loss = criterion(y_pred.view(-1, y_pred.shape[-1]), y.view(-1)) 
			# total_loss = final_label_loss + segment_loss

			val_losses.append(total_loss.item())

			prob, y_pred_indices = torch.max(final_output, dim=1)

			val_preds.append(y_pred_indices)
			val_targets.append(y_final)
				
		# calculate the accuracy
		train_preds = torch.cat(train_preds, dim=0)
		train_targets = torch.cat(train_targets, dim=0)
		train_acc = (train_preds == train_targets).float().mean().cpu()

		val_preds = torch.cat(val_preds, dim=0)
		val_targets = torch.cat(val_targets, dim=0)
		val_acc = (val_preds == val_targets).float().mean().cpu()

		train_acc_history.append(train_acc)
		val_acc_history.append(val_acc)

			# save the model checkpoint for the best validation accuracy
		if val_acc > max_val_acc:
			torch.save({'model_state_dict': model.state_dict()}, 'rna_type_checkpoint.pt')  
			best_epoch = epoch
			max_val_acc = val_acc

		train_loss_history.append(np.mean(train_losses))
		val_loss_history.append(np.mean(val_losses))

		# show intermediate steps
		if epoch % intermediate_evel == 0:
			tqdm.write(f'Epoch {epoch}/{epochs}: train loss={np.mean(train_loss_history):.6f}, '
							f'train acc={train_acc:.6f}, '
							f'val loss={np.mean(val_loss_history):.6f}, '
							f'val acc={val_acc:.6f}') 
			if early_stopper.early_stop(np.mean(val_loss_history)):  # Early stopper triggers when val_acc worse than the best one     
				tqdm.write("Early Stopping Trigger.") 
				tqdm.write(f"Best model at epoch {best_epoch} saved with val acc = {max_val_acc}")
				return train_result(train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_epoch)  
				
	tqdm.write(f"Best model at epoch {best_epoch} saved with val acc = {max_val_acc}")  
	
	return train_result(train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_epoch)

def test(x_test, y_test, view=0):

	test_dataset = RNATypeDataset(x_test, y_test)
	test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
	test_preds = []

	model = RNAClassifier(len(labels_ref), num_channels, kernel_size, dropout_rate, padding).to(device)
	model.load_state_dict(torch.load('rna_type_checkpoint.pt')['model_state_dict'])

	test_preds = []
	test_truth = []

	model.eval()
		
	for batch in test_loader:
		x, y = batch
		x, y = x.to(device).float(), y.to(device).long()

		# padding_token_count = []
		# for i in range(y.shape[0]):
		# 	c = 0
		# 	for j in range(y.shape[1]):
		# 		if y[i, j] == 15:
		# 			c += 1
		# 	c = 30 - c
		# 	padding_token_count.append(c)
		# padding_token_count = torch.tensor(padding_token_count, device=device, dtype=torch.float32)
		# get sequences label here
		y = y[:,0] # (16, )
		output, feature_maps = model(x)
		# Step 1: Sum across the second dimension (size 30) [16,16]
		final_output = torch.sum(output, dim=1)
		final_output = final_output[:,:-2]

		prob, y_pred = torch.max(final_output, dim=1)
		# prob_prediction_score = (prob / padding_token_count) * 100
		# prob_prediction_score = torch.clamp(prob_prediction_score, max=100)

		# print(prob, y_pred, padding_token_count, prob_prediction_score)
		test_preds.append(y_pred.cpu().numpy())	
		test_truth.append(y.cpu().numpy())	
	
	test_preds = np.concatenate(test_preds)
	test_truth = np.concatenate(test_truth)
	if view == 0: # overall performance
		result = calculate_metric_with_sklearn(test_truth, test_preds)
		print(result)
	else: # performance on seperate classes
		results_per_class = {}
		for i in range(len(labels_ref) - 2):
			result = calculate_binary_metrics(test_truth, test_preds, target_class=i)
			results_per_class[i] = result
			
	df = pd.DataFrame.from_dict(results_per_class, orient='index')
	df['Class Label'] = ["5S_rRNA", "5_8S_rRNA", "tRNA", "ribozyme", "CD-box", "miRNA", "Intron_gpI", "Intron_gpII", "scaRNA", "HACA-box", "riboswitch", "IRES", "leader", "mRNA"]
	df = df[['Class Label', 'Confusion Matrix', 'True Positives', 'True Negatives', 
		'False Positives', 'False Negatives', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']]
	numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']

	df[numeric_columns] = df[numeric_columns].round(3)

	df.to_csv("separate_class_result.csv", index=False)
	
	print(results_per_class)
	return


def test_one(sequence):

	sequence = ''.join({'T' : 'U'}.get(base, base) for base in sequence) 
	# print(sequence)
	data = [[("RNA1", sequence), 99]]
	crop_seq = crop_sequences(data)
	model, alphabet = fm.pretrained.rna_fm_t12()
	batch_converter = alphabet.get_batch_converter()
	model.to(device)
	# print(len(segments[0]))
	segments = []
	for i, seg in enumerate(crop_seq[0]):
		segments.append(seg[0])
	
	segment_tuple = []
	segment_batch_size = 30
	segment_length = 32

	lenght_wo_padding = []
	

	maximum_length = segment_batch_size * segment_length
	for k in range(0, len(segments), segment_batch_size):
		segment = ''.join(segments[k:k + segment_batch_size])
		lenght_wo_padding.append(len(segment) // segment_length)
		sequence_id = f"Sequence {i} Segment {k} to {min(k + segment_batch_size, len(segments)) - 1}"
		segment_padding = '-' * (maximum_length - len(segment))
		segment = segment + segment_padding
		segment_tuple.append((sequence_id, segment))
	
	batch_labels, batch_strs, batch_tokens = batch_converter(segment_tuple)
	
	with torch.no_grad():
		results = model(batch_tokens.to(device), repr_layers=[12])
		emb = results['representations'][12].cpu().numpy() 
	
	token_embeddings = []
	for j in range(emb.shape[0]):
		token_embeddings.append(emb[j:j+1])
	token_embeddings = np.concatenate(token_embeddings, axis=0)
	token_embeddings = token_embeddings[:, 1:-1, :]
	
	# print(token_embeddings.shape)
	temp = []
	mean_idx = [i for i in range(0, token_embeddings.shape[1], 32)]
	for i in range(token_embeddings.shape[0]):
		for j in range(len(mean_idx)):
			temp.append(np.mean(token_embeddings[i][mean_idx[j]: mean_idx[j]+32], axis=0))
		# Use the mean of the RNA-FM embedding across 32 items
	token_embeddings = torch.tensor(np.array(temp))
	x = token_embeddings.to(device).float()
	x = x.view(1, 30, 640)

	# label_data
	model = RNAClassifier(len(labels_ref), num_channels, kernel_size, dropout_rate, padding).to(device)
	model.load_state_dict(torch.load('rna_type_checkpoint.pt')['model_state_dict'])
	model.eval()
		
	y_pred, feature_map = model(x)  # y_pred: (1, 30, 15), feature_map: (1, 30, 640)
	final_output = torch.sum(y_pred, dim=1)
	final_output = final_output[:, :-2]  # Assuming you want to exclude the last 2 dimensions
	prob, y_p = torch.max(final_output, dim=1)
	print(y_p)

	# Select a specific class to analyze
	class_index = 1  # For example, class index 1

	# Get the probabilities for the specific class
	final_probs = y_pred[0, :, class_index]  # Shape: (30,)

	# Enable gradient tracking on the feature_map
	feature_map.requires_grad_()
	feature_map.retain_grad()  # Retain gradients for feature_map
	print(feature_map.shape)

	# Check if final_probs is valid before backward pass
	if final_probs is not None:
			print("Final probabilities:", final_probs)  # Debugging line
			final_probs.backward(torch.ones_like(final_probs))  # Backward pass
	else:
			print("final_probs is None.")

	# Get gradients for the feature_map
	gradients = feature_map.grad  # Shape: (1, 30, 640)

	# Check if gradients are computed
	if gradients is not None:
			print(gradients.shape)
	else:
			print("Gradients are None. Check the backward pass.")
			exit()
	# Compute saliency maps for each channel
	positive_saliency = gradients.clamp(min=0)
	saliency_mean = positive_saliency.mean(dim=2)  # Average over features
	
	# Normalize the saliency values for visualization
	saliency_normalized = saliency_mean / saliency_mean.max()

	# Visualize the saliency map
	# Convert to numpy for visualization (if needed)
	saliency_normalized_np = saliency_normalized.squeeze().cpu().numpy()  # Shape: (30,)
	saliency_normalized_np = saliency_normalized_np.reshape(1, 30)  # Shape: (1, 30)

	# Create a new figure with a specified size
	plt.figure(figsize=(10, 3))  # Adjust the height to increase y-axis space

	# Display the heatmap
	plt.imshow(saliency_normalized_np, cmap='hot', aspect='auto', extent=[0, 30, 0, 1])  # Set extent for clarity
	plt.colorbar(label='Saliency Score')
	# Remove ylim or set it to fit better
	
	plt.title(f'Saliency Map for Class {class_index}')
	plt.xlabel('Segments')
	plt.ylabel('Saliency Score')
	plt.savefig('importance1.png')

	return


if __name__ == "__main__":
	# Extract data from files and label them all
	data_dir = [glob.glob(r'./data/fa_train*')[0], glob.glob(r'./data/fa_test*')[0], glob.glob(r'./data/fa_val*')[0]]
	sequences_labels_pairs  = label_data(data_dir)
	
	# Declare embedding model
	embedding_model, alphabet = fm.pretrained.rna_fm_t12()
	batch_converter = alphabet.get_batch_converter()

	embedding_model.to(device)
	embedding_model.eval()

	# Preprocess the data, crop it and apply embedding calculation here
	data = [crop_sequences(sequences_labels_pairs[i]) for i in range(3)]
	# Given segment_length=32, overlap=10, (N, 34, 640) for each sequence, train/test/val data -> N ~ 50
	# x_val, y_val, val_str, val_idx = generate_embed(data[2], embedding_model, 32)
	# print(x_val.shape, y_val.shape, val_str.shape, val_idx.shape) # (741, 964, 640) (741, 30) (741,) (700, 2)
	# x_train, y_train, train_str, train_idx = generate_embed(data[0], embedding_model, 32)
	# print(x_train.shape, y_train.shape, train_str.shape) 
	x_test, y_test, test_str, test_idx = generate_embed(data[1], embedding_model, 32)
	print(x_test.shape, y_test.shape, test_str.shape)

	# train model with constructed embedding
	# train(x_train, y_train, x_val, y_val) 
	test(x_test, y_test, 1)
	# test_one("GACTCTCGGCAACGGATATCTCGACTCTCGCATCGATGAAGAAAGTAGCAAAATGCGATACGTGGTGTGAATTGGACAATCCCGTGAATCGTCGAATCTTTGAACGCAAGTTGCGCCGAAGCCTTCCGACCGGGGGCACGTCTGCTTGGGCGTTA")



#def test_kf_no_training(x, y, num_folds=10, batch_size=16):
	
# 	skf = StratifiedKFold(n_splits=num_folds)
# 	fold_accuracies = []  # List to store accuracy for each fold

# 	# Load the pre-trained model once
# 	model = RNAClassifier(len(labels_ref), num_channels, kernel_size, dropout_rate).to(device)
# 	model.load_state_dict(torch.load('rna_type_checkpoint.pt')['model_state_dict'])
# 	model.eval()  # Set the model to evaluation mode

# 	for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
# 		print(f'Fold {fold + 1}/{num_folds}')

# 		x_train, x_val = x[train_index], x[val_index]
# 		y_train, y_val = y[train_index], y[val_index]
	
# 		val_dataset = RNATypeDataset(x_val, y_val)
# 		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 		val_preds = []
# 		val_targets = []

# 		with torch.no_grad():
# 			for batch in val_loader:
# 				x_val_batch, y_val_batch = batch
# 				x_val_batch, y_val_batch = x_val_batch.to(device).float(), y_val_batch.to(device).long()
				
# 				output, _ = model(x_val_batch)
# 				_, y_pred = torch.max(output, 2)
# 				val_preds.append(y_pred.cpu().numpy())
# 				val_targets.append(y_val_batch.cpu().numpy())

# 			val_preds = np.concatenate(val_preds)
# 			val_targets = np.concatenate(val_targets)

# 			val_acc = (val_preds == val_targets).mean()
# 			fold_accuracies.append(val_acc)  # Store the accuracy of the current fold
# 			print(f'Validation Accuracy for Fold {fold + 1}: {val_acc:.4f}')
	
# 	average_accuracy = np.mean(fold_accuracies)
# 	print(f'Average Validation Accuracy over {num_folds} folds: {average_accuracy:.4f}')
	
# 	return