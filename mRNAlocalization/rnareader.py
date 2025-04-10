# import RNA-FM LLM
# define the classifer layer (7 binary problems with OvR)
import torch
import numpy as np
import fm
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
import os

from gene_data import Genedata
from hier_attention_mask import AttentionMask
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/"
DATA_FILE = DATA_PATH + "modified_multilabel_seq_nonredundent.fasta"
#DATA_FILE = DATA_PATH + "test.fasta"


encoding_seq = OrderedDict([
	('A', [1, 0, 0, 0]),
	('C', [0, 1, 0, 0]),
	('G', [0, 0, 1, 0]),
	('T', [0, 0, 0, 1]),
	('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
	('-', [0, 0, 0, 0]),  # Pad
])

# Load mRNA-FM model (LLM)
class mRNA_FM:
	def __init__(self):
		embedding_model, alphabet = fm.pretrained.mrna_fm_t12()
		batch_converter = alphabet.get_batch_converter()
		embedding_model.eval()  # disables dropout for deterministic results
		self.model = embedding_model
		self.batch_converter = batch_converter
		self.max_length = 1022 * 3  # Maximum sequence length for rna_fm

	def embeddings(self, data):
		"""
		Generate embeddings for sequences, handling sequences longer than max_length.
		"""
		all_embeddings = []

		# Ensure the model is on the GPU
		self.model = self.model.to(DEVICE)

		for label, sequence in tqdm(data, desc="Generating embeddings", unit="sequence"):
			# Split sequence into chunks of max_length, ensuring divisibility by 3
			chunks = [
				sequence[i:i + self.max_length]
				for i in range(0, len(sequence), self.max_length)
			]
			# Ensure each chunk is divisible by 3
			chunks = [chunk[:len(chunk) - (len(chunk) % 3)] for chunk in chunks]
		
			chunk_embeddings = []
			chunk_lengths = []  # Store the lengths of each chunk
		
			for chunk in chunks:
				batch_data = [(label, chunk)]
				batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
		
				batch_tokens = batch_tokens.to(DEVICE)
		
				with torch.no_grad():
					results = self.model(batch_tokens, repr_layers=[12])
				chunk_embedding = results["representations"][12]  # Shape: [1, seq_len, 1280]
		
				pooled_embedding = torch.mean(chunk_embedding, dim=1)  # Shape: [1, 1280]
				chunk_embeddings.append(pooled_embedding)
		
				# Store the length of the chunk
				chunk_lengths.append(chunk_embedding.shape[1])  # seq_len
		
			total_length = sum(chunk_lengths)
			weights = [length / total_length for length in chunk_lengths]
			# Total length = 1020 + 1020 + 500 = 2540
			# Weights = [1020/2540, 1020/2540, 500/2540] ≈ [0.4016, 0.4016, 0.1968]
			# combined_embedding = 0.4016 * embedding1 + 0.4016 * embedding2 + 0.1968 * embedding3
			combined_embedding = sum(weight * embedding for weight, embedding in zip(weights, chunk_embeddings))  # Shape: [1, 1280]
		
			all_embeddings.append(combined_embedding)

		return torch.cat(all_embeddings, dim=0)  # Shape: [batch_size, 1280]
		
class LLMClassifier(nn.Module):
	def __init__(self, output_dim):
		super(LLMClassifier, self).__init__()
		self.output_dim = output_dim

		self.fc1 = nn.Linear(1280, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, output_dim)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		x = self.fc1(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc3(x)
		return x

class MultiscaleCNNLayers(nn.Module):
	def __init__(self, in_channels, embedding_dim, pooling_size, pooling_stride, drop_rate_cnn, drop_rate_fc, nb_classes):
		super(MultiscaleCNNLayers, self).__init__()

		self.bn1 = nn.BatchNorm1d(in_channels)
		self.bn2 = nn.BatchNorm1d(in_channels // 2)

		self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels, kernel_size=9, padding="same")
		self.conv1_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=9, padding="same")
		self.conv2_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels, kernel_size=20, padding="same")
		self.conv2_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=20, padding="same")
		self.conv3_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels // 2, kernel_size=49, padding="same")
		self.conv3_2 = nn.Conv1d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=49, padding="same")

		self.pool = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_stride)
	
		self.dropout_cnn = nn.Dropout(drop_rate_cnn)
		self.dropout_fc = nn.Dropout(drop_rate_fc)

		self.fc = nn.Linear(999, nb_classes)
		self.activation = nn.GELU() 

	def forward_cnn(self, x, conv1, conv2, bn1, bn2):
		x = conv1(x)
		x = self.activation(bn1(x))
		x = conv2(x)
		x = self.activation(bn2(x))
		x = self.pool(x)
		x = self.dropout_cnn(x)
		return x
	
class MultiscaleCNNModel(nn.Module):
	def __init__(self, layers):
		super(MultiscaleCNNModel, self).__init__()
		self.layers = layers

	def forward(self, x):
		x1 = self.layers.forward_cnn(x, self.layers.conv1_1, self.layers.conv1_2, self.layers.bn1, self.layers.bn2)
		x2 = self.layers.forward_cnn(x, self.layers.conv2_1, self.layers.conv2_2, self.layers.bn1, self.layers.bn2)
		x3 = self.layers.forward_cnn(x, self.layers.conv3_1, self.layers.conv3_2, self.layers.bn2, self.layers.bn2)
		x = torch.cat((x1, x2, x3), dim=1)
		x = self.layers.dropout_fc(self.layers.fc(x))
		return x

class EnsembleModel(nn.Module):
	def __init__(self, llm_model, cnn_model, llm_output_dim, cnn_output_dim, hidden_dim, nb_classes):
		super(EnsembleModel, self).__init__()
		self.llm_model = llm_model
		self.cnn_model = cnn_model

		# Fully connected NN for combining LLM and CNN outputs and length compoent in second layer
		self.fc1 = nn.Linear(llm_output_dim + cnn_output_dim, (llm_output_dim + cnn_output_dim)/ 2 + 1)
		self.fc2 = nn.Linear((llm_output_dim + cnn_output_dim)/ 2 + 1, nb_classes)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.2)
	
	def forward(self, x):
		llm_output = self.llm_model(x) # LLM output
		cnn_output = self.cnn_model(x) # CNN output

		x = torch.cat((llm_output, cnn_output), dim=1) # Combine LLM and CNN outputs

		x = self.fc1(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x

class GeneDataset(Dataset):
	def __init__(self, sequences, labels):
		self.sequences = sequences
		self.labels = labels

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		return self.sequences[idx], self.labels[idx]
	
def get_id_label_seq_Dict(gene_data):
#	{
#   	gene_id_1: {label_1: (seqLeft_1, seqRight_1)},
#    	gene_id_2: {label_2: (seqLeft_2, seqRight_2)}, ...
#	}
	id_label_seq_Dict = OrderedDict()
	for gene in gene_data:
		label = gene.label
		gene_id = gene.id.strip()
		id_label_seq_Dict[gene_id] = {}
		id_label_seq_Dict[gene_id][label]= (gene.seqLeft, gene.seqRight)

	return id_label_seq_Dict

def get_label_id_Dict(id_label_seq_Dict):
#	{
#   	label_1: {gene_id_1, gene_id_2, ...},
#    	label_2: {gene_id_3, gene_id_4}, ...
#	}
	label_id_Dict = OrderedDict()
	for eachkey in id_label_seq_Dict.keys():
		label = list(id_label_seq_Dict[eachkey].keys())[0]
		label_id_Dict.setdefault(label,set()).add(eachkey)
	
	return label_id_Dict

def group_sample(label_id_Dict,datasetfolder,foldnum=8):
	Train = OrderedDict()
	Test = OrderedDict()
	Val = OrderedDict()
	for i in range(foldnum):
		Train.setdefault(i,list())
		Test.setdefault(i,list())
		Val.setdefault(i,list())
	
	for eachkey in label_id_Dict:
		label_ids = list(label_id_Dict[eachkey])
		if len(label_ids)<foldnum:
			for i in range(foldnum):
				Train[i].extend(label_ids)
			continue
		
		[train_fold_ids, val_fold_ids,test_fold_ids] = KFoldSampling(label_ids, foldnum)
		for i in range(foldnum):
			Train[i].extend(train_fold_ids[i])
			Val[i].extend(val_fold_ids[i])
			Test[i].extend(test_fold_ids[i])
			#print('label:%s finished sampling! Train length: %s, Test length: %s, Val length:%s'%(eachkey, len(train_fold_ids[i]), len(test_fold_ids[i]),len(val_fold_ids[i])))
	
	for i in range(foldnum):
		print('Train length: %s, Test length: %s, Val length: %s'%(len(Train[i]),len(Test[i]),len(Val[i])))
		#print(type(Train[i]))
		#print(Train[0][:foldnum])
		np.savetxt(datasetfolder+'/Train'+str(i)+'.txt', np.asarray(Train[i]),fmt="%s")
		np.savetxt(datasetfolder+'/Test'+str(i)+'.txt', np.asarray(Test[i]),fmt="%s")
		np.savetxt(datasetfolder+'/Val'+str(i)+'.txt', np.asarray(Val[i]),fmt="%s")
	
	return Train, Test, Val

def KFoldSampling(ids, k):
	kf = KFold(n_splits=k, shuffle=True, random_state=1234)
	folds = kf.split(ids)
	train_fold_ids = OrderedDict()
	val_fold_ids = OrderedDict()
	test_fold_ids = OrderedDict()
	for i, (train_indices, test_indices) in enumerate(folds):
		size_all = len(train_indices)
		train_fold_ids[i] = []
		val_fold_ids[i] = []
		test_fold_ids[i] = []
		train_indices2 = train_indices[:int(size_all * 0.8)]
		val_indices = train_indices[int(size_all * 0.8):]
		for s in train_indices2:
			train_fold_ids[i].append(ids[s])

		for s in val_indices:
			val_fold_ids[i].append(ids[s])

		for s in test_indices:
			test_fold_ids[i].append(ids[s])

	return train_fold_ids, val_fold_ids, test_fold_ids

def label_dist(dist):
	return [int(x) for x in dist]

def preprocess_data_onehot(left=3999, right=3999, k_fold=8):
	# Prepare data
	data = Genedata.load_sequence(
		dataset=DATA_FILE,
		left=left, # divsible by 3
		right=right,
		predict=False,
	)
	id_label_seq_dict = get_id_label_seq_Dict(data)
	label_id_dict = get_label_id_Dict(id_label_seq_dict)
	Train, Test, Val = group_sample(label_id_dict, DATA_PATH, k_fold)

	X_train, X_test, X_val = {}, {}, {}
	Y_train, Y_test, Y_val = {}, {}, {}
	for i in tqdm(range(len(Train))):  # Fold num
		tqdm.write(f"Padding and Indexing data for fold {i+1} (One-Hot Encoding)")
		seq_encoding_keys = list(encoding_seq.keys())
		seq_encoding_vectors = np.array(list(encoding_seq.values()))

		# Train data
		X_train[i] = []
		for id in Train[i]:
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			# One-hot encode
			one_hot_left = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_left]
			one_hot_right = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_right]
			# Combine left and right
			combined = np.concatenate([one_hot_left, one_hot_right], axis=0)
			X_train[i].append(combined)

		# Convert list of NumPy arrays to a single NumPy array, then to a PyTorch tensor
		X_train[i] = torch.tensor(np.array(X_train[i]), dtype=torch.float32).permute(0, 2, 1)
		Y_train[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Train[i]], dtype=torch.long)

		# Test data
		X_test[i] = []
		for id in Test[i]:
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			# One-hot encode
			one_hot_left = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_left]
			one_hot_right = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_right]
			# Combine left and right
			combined = np.concatenate([one_hot_left, one_hot_right], axis=0)
			X_test[i].append(combined)

		# Convert list of NumPy arrays to a single NumPy array, then to a PyTorch tensor
		X_test[i] = torch.tensor(np.array(X_test[i]), dtype=torch.float32).permute(0, 2, 1)
		Y_test[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Test[i]], dtype=torch.long)

		# Validation data
		X_val[i] = []
		for id in Val[i]:
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			# One-hot encode
			one_hot_left = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_left]
			one_hot_right = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_right]
			# Combine left and right
			combined = np.concatenate([one_hot_left, one_hot_right], axis=0)
			X_val[i].append(combined)

		# Convert list of NumPy arrays to a single NumPy array, then to a PyTorch tensor
		X_val[i] = torch.tensor(np.array(X_val[i]), dtype=torch.float32).permute(0, 2, 1)
		Y_val[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Val[i]], dtype=torch.long)
	
	return X_train, X_test, X_val, Y_train, Y_test, Y_val

def preprocess_data_raw_with_embeddings(left=3999, right=3999):
    """
    Prepare raw data for LLM training and generate embeddings using mRNA_FM.
    """
    # Load raw data
    data = Genedata.load_sequence(
        dataset=DATA_FILE,
        left=left,  # Divisible by 3
        right=right,
        predict=False,
    )
    id_label_seq_dict = get_id_label_seq_Dict(data)
    label_id_dict = get_label_id_Dict(id_label_seq_dict)
    Train, Test, Val = group_sample(label_id_dict, DATA_PATH)

    X_train, X_test, X_val = {}, {}, {}
    Y_train, Y_test, Y_val = {}, {}, {}

    # Initialize mRNA_FM model
    mrna_fm = mRNA_FM()

    for i in tqdm(range(len(Train))):  # Fold num
        tqdm.write(f"Preparing raw data and generating embeddings for fold {i+1}")

        # Train data
        train_data = [
            (
                "id_" + str(idx),
                (list(id_label_seq_dict[id].values())[0][0] + list(id_label_seq_dict[id].values())[0][1])
            )
            for idx, id in enumerate(Train[i])
        ]
        X_train[i] = torch.tensor(mrna_fm.embeddings(train_data).cpu().numpy(), dtype=torch.float32)  # Convert to Tensor
        Y_train[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Train[i]], dtype=torch.float32)

        # Test data
        test_data = [
            (
                "id_" + str(idx),
                (list(id_label_seq_dict[id].values())[0][0] + list(id_label_seq_dict[id].values())[0][1])
            )
            for idx, id in enumerate(Test[i])
        ]
        X_test[i] = torch.tensor(mrna_fm.embeddings(test_data).cpu().numpy(), dtype=torch.float32)  # Convert to Tensor
        Y_test[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Test[i]], dtype=torch.float32)

        # Validation data
        val_data = [
            (
                "id_" + str(idx),
                (list(id_label_seq_dict[id].values())[0][0] + list(id_label_seq_dict[id].values())[0][1])
            )
            for idx, id in enumerate(Val[i])
        ]
        X_val[i] = torch.tensor(mrna_fm.embeddings(val_data).cpu().numpy(), dtype=torch.float32)  # Convert to Tensor
        Y_val[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Val[i]], dtype=torch.float32)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

def train_model(model, name, X_train, Y_train, X_test, Y_test, X_val,
			Y_val, batch_size, epochs=50, lr=1e-5, save_path="./models", log_file="training_log.txt"):
	""" Train the LLM/CNN model and write logs to a file """
	model = model.to(DEVICE)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	# Open the log file
	with open(log_file, "w") as log:
		for i in tqdm(range(len(X_train))):  # Fold num
			log.write(f"fold {i+1} ({name} Training)\n")
			tqdm.write(f"fold {i+1} ({name} Training)")

			train_dataset = GeneDataset(X_train[i], Y_train[i])
			val_dataset = GeneDataset(X_val[i], Y_val[i])
			test_dataset = GeneDataset(X_test[i], Y_test[i])
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
			test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

			best_val_loss = float('inf')

			for epoch in range(epochs):
				# Training step
				model.train()
				train_loss = 0
				for sequences, labels in train_loader:
					sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
					#labels = labels.long()
					optimizer.zero_grad()
					outputs = model(sequences)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
					train_loss += loss.item()

				train_loss /= len(train_loader)

				# Validation step
				model.eval()
				with torch.no_grad():
					val_loss = 0
					for sequences, labels in val_loader:
						sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
						outputs = model(sequences)
						loss = criterion(outputs, labels)
						val_loss += loss.item()

				val_loss /= len(val_loader)
				
				log_message = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
				log.write(log_message)
				tqdm.write(log_message)

				# Save the best model
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					model_save_path = os.path.join(save_path, f"{name}_model_fold{i+1}.pth")
					torch.save(model.state_dict(), model_save_path)
					log.write(f"Best model for fold {i+1} saved with Val Loss: {best_val_loss:.4f}\n")
					tqdm.write(f"Best model for fold {i+1} saved with Val Loss: {best_val_loss:.4f}")

			# Load the best model for this fold
			best_model_path = os.path.join(save_path, f"{name}_model_fold{i+1}.pth")
			model.load_state_dict(torch.load(best_model_path))
			log.write(f"Best model for fold {i+1} loaded from {best_model_path}\n")
			tqdm.write(f"Best model for fold {i+1} loaded from {best_model_path}")

			# Test step
			model.eval()
			with torch.no_grad():
				test_loss = 0
				for sequences, labels in test_loader:
					sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
					outputs = model(sequences)
					loss = criterion(outputs, labels)
					test_loss += loss.item()

			test_loss /= len(test_loader)
			log.write(f"Test Loss: {test_loss:.4f}\n")
			tqdm.write(f"Test Loss: {test_loss:.4f}")
	return


if __name__ == "__main__":
	# Load data for CNN
	X_train_cnn, X_test_cnn, X_val_cnn, Y_train_cnn, Y_test_cnn, Y_val_cnn = preprocess_data_onehot(
		left=3999,
		right=3999,
		k_fold=8
	)
	print(X_train_cnn[0].shape, Y_train_cnn[0].shape) #(12093, 4, 7998) (12093, 7)
	print(X_test_cnn[0].shape, Y_test_cnn[0].shape)
	print(X_val_cnn[0].shape, Y_val_cnn[0].shape)

	# Load data for LLM
	X_train_llm, X_test_llm, X_val_llm, Y_train_llm, Y_test_llm, Y_val_llm = preprocess_data_raw_with_embeddings(
		left=3999,
		right=3999
	)
	print(X_train_llm[0].shape, Y_train_llm[0].shape) 
	print(X_test_llm[0].shape, Y_test_llm[0].shape) 
	print(X_val_llm[0].shape, Y_val_llm[0].shape) 

	### Initialize CNN model
	cnn_layers = MultiscaleCNNLayers(
	    in_channels=64,
	    embedding_dim=4,  # For one-hot encoding
	    pooling_size=8,
	    pooling_stride=8,
	    drop_rate_cnn=0.2,
	    drop_rate_fc=0.2,
	    nb_classes=7
	)
	cnn_model = MultiscaleCNNModel(cnn_layers).to(DEVICE)

	# Train CNN model
	train_model(
		model=cnn_model,
		name="CNN",
		X_train=X_train_cnn,
		Y_train=Y_train_cnn,
		X_test=X_test_cnn,
		Y_test=Y_test_cnn,
		X_val=X_val_cnn,
		Y_val=Y_val_cnn,
		batch_size=32,
		epochs=10,
		lr=1e-5,
		save_path="./cnn_models",
		log_file="cnn_training_log.txt"
	)

	# Initialize LLM model
	llm_model = LLMClassifier(
		output_dim=7
	).to(DEVICE)

	##print(llm_model)
	## Train LLM model
	train_model(
		model=llm_model,
		name="LLM",
		X_train=X_train_llm,
		Y_train=Y_train_llm,
		X_test=X_test_llm,
		Y_test=Y_test_llm,
		X_val=X_val_llm,
		Y_val=Y_val_llm,
		batch_size=32,
		epochs=10,
		lr=1e-5,
		save_path="./llm_models",
		log_file="llm_training_log.txt"
	)

	# Initialize Ensemble model
	#ensemble_model = EnsembleModel(
	#    llm_model=llm_model,
	#    cnn_model=cnn_model,
	#    llm_output_dim=768,  # Output dimension of LLM
	#    cnn_output_dim=6,    # Output dimension of CNN
	#    nb_classes=6
	#).to(DEVICE)

	# Train Ensemble model (implement a new function for this)
	# train_ensemble_model(...)
# combine LLM and CNN part with single NN for final anwser]
	#model, alphabet = fm.pretrained.mrna_fm_t12()
	#batch_converter = alphabet.get_batch_converter()
	#model.eval()  # disables dropout for deterministic results

	## Prepare data
	#data = [
	#	("CDS2", "AAGAUAAAGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGAUGAUGAUGAUGAUGAUGAUGAUGAUGAUGAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGAUGAUGAUGAUGAUGAUGAUGAUGAUGAUGAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCAAUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
	#]
	#batch_labels, batch_strs, batch_tokens = batch_converter(data)

	## Extract embeddings (on CPU)
	#with torch.no_grad():
	#	results = model(batch_tokens, repr_layers=[12])
	#token_embeddings = results["representations"][12]
	#print(token_embeddings.shape)  # Shape: [batch_size, seq_len, embedding_dim]