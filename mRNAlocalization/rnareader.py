# import RNA-FM LLM
# define the classifer layer (7 binary problems with OvR)
import torch
import numpy as np
import fm
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from gene_data import Genedata
from hier_attention_mask import AttentionMask
from sklearn.model_selection import KFold, StratifiedKFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/"
DATA_FILE = DATA_PATH + "modified_multilabel_seq_nonredundent.fasta"

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

	def embeddings(self, data):
		batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
		with torch.no_grad():
			results = self.model(batch_tokens, repr_layers=[12])
		token_embeddings = results["representations"][12]
		return token_embeddings
	
class LLMClassifier(nn.Module):
	def __init__(self, llm_model, output_dim):
		super(LLMClassifier, self).__init__()
		self.llm_model = mRNA_FM()
		self.output_dim = output_dim

		self.fc1 = nn.Linear(768, 512)
		self.fc2 = nn.Linear(512, output_dim)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		x = self.llm_model(x) # embedding model
		x = self.fc1(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc2(x)
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

		self.fc = nn.Linear(100, nb_classes)
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
		x3 = self.layers.forward_cnn(x, self.layers.conv3_1, self.layers.conv3_2, self.layers.bn1, self.layers.bn2)
		x = torch.cat((x1, x2, x3), dim=1)
		x = self.layers.dropout_fc(self.layers.fc(x))
		return x

class EnsembleModel(nn.Module):
	def __init__(self, llm_model, cnn_model, llm_output_dim, cnn_output_dim, hidden_dim, nb_classes):
		super(EnsembleModel, self).__init__()
		self.llm_model = llm_model
		self.cnn_model = cnn_model

		# Fully connected NN for combining LLM and CNN outputs
		self.fc1 = nn.Linear(llm_output_dim + cnn_output_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, nb_classes)
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

def preprocess_data_onehot(left=3999, right=3999, pooling_size=3):
	# Prepare data
	data = Genedata.load_sequence(
		dataset=DATA_FILE,
		left=left, # divsible by 3
		right=right,
		predict=False,
	)
	id_label_seq_dict = get_id_label_seq_Dict(data)
	label_id_dict = get_label_id_Dict(id_label_seq_dict)
	Train, Test, Val = group_sample(label_id_dict, DATA_PATH)

	X_train, X_test, X_val = {}, {}, {}
	Y_train, Y_test, Y_val = {}, {}, {}

	for i in tqdm(range(len(Train))): # fold num
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

		X_train[i] = np.array(X_train[i])
		Y_train[i] = np.array([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Train[i]])

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

		X_test[i] = np.array(X_test[i])
		Y_test[i] = np.array([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Test[i]])

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

		X_val[i] = np.array(X_val[i])
		Y_val[i] = np.array([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Val[i]])

		#print(X_train[0].shape, Y_train[0].shape) #(12093, 7998, 4) (12093, 7)
	return X_train, X_test, X_val, Y_train, Y_test, Y_val


if __name__ == "__main__":

	# Load data
	X_train, X_test, X_val, Y_train, Y_test, Y_val = preprocess_data_onehot(
		left=3999,
		right=3999,
		pooling_size=3
	)

	print("X_train shape:", X_train[0].shape)
	print("Y_train shape:", Y_train[0].shape)
	print("X_test shape:", X_test[0].shape)
	print("Y_test shape:", Y_test[0].shape)	
	print("X_val shape:", X_val[0].shape)
	print("Y_val shape:", Y_val[0].shape)
	
	# Convert data to torch tensors


	#ensemble_model = ensemble_model.to(DEVICE)
	#X_train_tensor = torch.tensor(X_train[0], dtype=torch.float32).to(DEVICE)
	#X_test_tensor = torch.tensor(X_test[0], dtype=torch.float32).to(DEVICE)
	#X_val_tensor = torch.tensor(X_val[0], dtype=torch.float32).to(DEVICE)
	#Y_train_tensor = torch.tensor(Y_train[0], dtype=torch.float32).to(DEVICE)
	#Y_test_tensor = torch.tensor(Y_test[0], dtype=torch.float32).to(DEVICE)
	#Y_val_tensor = torch.tensor(Y_val[0], dtype=torch.float32).to(DEVICE)
	
	# Load mRNA-FM model, LLM class
	llm_model = LLMClassifier(
		llm_model=mRNA_FM(),
		output_dim=6
	)
	# llm_output = llm_model(X_train_tensor)
	
	#print("Token embeddings shape:", llm_token_embeddings.shape)
	#print("Token embeddings:", llm_token_embeddings)
	
	# one hot 
	cnn_model = MultiscaleCNNLayers(
		in_channels=64,
		embedding_dim=4,  # For one-hot encoding
		pooling_size=8,
		pooling_stride=8,
		drop_rate_cnn=0.2,
		drop_rate_fc=0.2,
		nb_classes=6
	)
	# cnn_output = llm_model(X_train_tensor)


	ensemble_model = EnsembleModel(
		llm_model=llm_model,
		cnn_model=cnn_model,
		llm_output_dim=768,  
		cnn_output_dim=6,
		hidden_dim=128,
		nb_classes=6
	)
	
	output = ensemble_model(llm_output, cnn_output)

	# Output of ensemble_model code
	


	# token_embeddings: (batch_size, seq_len, embedding_dim)



# combine LLM and CNN part with single NN for final anwser