import datetime
import itertools
from collections import OrderedDict
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
basedir='./'
sys.path.append(basedir)
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from multihead_attention_model import MultiheadAttentionModel
from gene_data import Gene_data

device = "cuda" if torch.cuda.is_available() else "cpu"
encoding_seq = OrderedDict([
	('UNK', [0, 0, 0, 0]),
	('A', [1, 0, 0, 0]),
	('C', [0, 1, 0, 0]),
	('G', [0, 0, 1, 0]),
	('T', [0, 0, 0, 1]),
	('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))

gene_ids = None

def calculating_class_weights(y_true):
	number_dim = np.shape(y_true)[1]
	weights = np.empty([number_dim, 2])
	for i in range(number_dim):
		weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
	return weights


def get_id_label_seq_Dict(gene_data):
	id_label_seq_Dict = OrderedDict()
	for gene in gene_data:
		label = gene.label
		gene_id = gene.id.strip()
		id_label_seq_Dict[gene_id] = {}
		id_label_seq_Dict[gene_id][label]= (gene.seqleft,gene.seqright)
	
	return id_label_seq_Dict


def get_label_id_Dict(id_label_seq_Dict):
	label_id_Dict = OrderedDict()
	for eachkey in id_label_seq_Dict.keys():
		label = list(id_label_seq_Dict[eachkey].keys())[0]
		label_id_Dict.setdefault(label,set()).add(eachkey)
	
	return label_id_Dict

def typeicalSampling(ids, k):
	kf = KFold(n_splits=k, shuffle=True, random_state=1234)
	folds = kf.split(ids)
	train_fold_ids = OrderedDict()
	val_fold_ids = OrderedDict()
	test_fold_ids=OrderedDict()
	for i, (train_indices, test_indices) in enumerate(folds):
		size_all = len(train_indices)
		train_fold_ids[i] = []
		val_fold_ids[i] = []
		test_fold_ids[i]  =[]
		train_indices2 = train_indices[:int(size_all * 0.8)]
		val_indices = train_indices[int(size_all * 0.8):]
		for s in train_indices2:
			train_fold_ids[i].append(ids[s])
		
		for s in val_indices:
			val_fold_ids[i].append(ids[s])
		
		for s in test_indices:
			test_fold_ids[i].append(ids[s])
		

	return train_fold_ids,val_fold_ids,test_fold_ids

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
		
		[train_fold_ids, val_fold_ids,test_fold_ids] = typeicalSampling(label_ids, foldnum)
		for i in range(foldnum):
			Train[i].extend(train_fold_ids[i])
			Val[i].extend(val_fold_ids[i])
			Test[i].extend(test_fold_ids[i])
			print('label:%s finished sampling! Train length: %s, Test length: %s, Val length:%s'%(eachkey, len(train_fold_ids[i]), len(test_fold_ids[i]),len(val_fold_ids[i])))
	
	for i in range(foldnum):
		print('Train length: %s, Test length: %s, Val length: %s'%(len(Train[i]),len(Test[i]),len(Val[i])))
		#print(type(Train[i]))
		#print(Train[0][:foldnum])
		np.savetxt(datasetfolder+'/Train8'+str(i)+'.txt', np.asarray(Train[i]),fmt="%s")
		np.savetxt(datasetfolder+'/Test8'+str(i)+'.txt', np.asarray(Test[i]),fmt="%s")
		np.savetxt(datasetfolder+'/Val8'+str(i)+'.txt', np.asarray(Val[i]),fmt="%s")
	
	return Train, Test, Val

def label_dist(dist):
	#assert (len(dist) == 4)
	return [int(x) for x in dist]

def maxpooling_mask(input_mask,pool_length=3):
	#input_mask is [N,length]
	max_index = int(input_mask.shape[1]/pool_length)-1
	max_all=np.zeros([input_mask.shape[0],int(input_mask.shape[1]/pool_length)])
	for i in range(len(input_mask)):
		index=0
		for j in range(0,len(input_mask[i]),pool_length):
			if index<=max_index:
				max_all[i,index] = np.max(input_mask[i,j:(j+pool_length)])
				index+=1
	
	return max_all


def preprocess_data(left, right, dataset, padmod='center', pooling_size=3): # edit needed
	gene_data = Gene_data.load_sequence(dataset, left, right)
	id_label_seq_Dict = get_id_label_seq_Dict(gene_data)
	label_id_Dict = get_label_id_Dict(id_label_seq_Dict)
	Train=OrderedDict()
	Test=OrderedDict()
	Val=OrderedDict()
	datasetfolder=os.path.dirname(dataset)
	if os.path.exists(datasetfolder+'/Train8'+str(0)+'.txt'):
		for i in range(8):
			Train[i] = np.loadtxt(datasetfolder+'/Train8'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Train')[:]
			Test[i] = np.loadtxt(datasetfolder+'/Test8'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Test')[:]
			Val[i] = np.loadtxt(datasetfolder+'/Val8'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Val')[:]
	else:
		[Train, Test,Val] = group_sample(label_id_Dict,datasetfolder)
	
	Xtrain={}
	Xtest={}
	Xval={}
	Ytrain={}
	Ytest={}
	Yval={}
	Train_mask_label={}
	Test_mask_label={}
	Val_mask_label={}
	maxpoolingmax = int((left+right)/pooling_size)
	
	for i in range(8):
		#if i <2:
		#   continue
		
		print('padding and indexing data')
		encoding_keys = seq_encoding_keys
		encoding_vectors = seq_encoding_vectors
		#train
		#padd center
		X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Train[i]]
		X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Train[i]]
		if padmod =='center':
			mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
			mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
			mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
			Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
			X_left = pad_sequences(X_left,maxlen=left,
							   dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
			
			X_right = pad_sequences(X_right,maxlen=right,
						  dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
			
			Xtrain[i] = np.concatenate([X_left,X_right],axis = -1)
		else:
		   #merge left and right and padding after sequence
			Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
			Xtrain[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
			#mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
			#Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
			Train_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
		
		Ytrain[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Train[i]])
		print("training shapes"+str(Xtrain[i].shape)+" "+str(Ytrain[i].shape))
		
		#test
		X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Test[i]]
		X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Test[i]]
		if padmod =='center':
			mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
			mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
			mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
			Test_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
			X_left = pad_sequences(X_left,maxlen=left,
							   dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
			
			X_right = pad_sequences(X_right,maxlen=right,
						  dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
			
			Xtest[i] = np.concatenate([X_left,X_right],axis = -1)
		else:
			#merge left and right and padding after sequence
			Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
			Xtest[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
			#mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
			#Test_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
			Test_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
		
		Ytest[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Test[i]])
		#validation
		X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Val[i]]
		X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Val[i]]
		if padmod=='center':
			mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
			mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
			mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
			Val_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
			X_left = pad_sequences(X_left,maxlen=left,
							   dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
			
			X_right = pad_sequences(X_right,maxlen=right,
						  dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
			
			Xval[i] = np.concatenate([X_left,X_right],axis = -1)
		else:
			#merge left and right and padding after sequence
			Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
			Xval[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
			#mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
			#Val_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
			Val_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
		
		Yval[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Val[i]])
	
	return Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors

class GeneDataset(Dataset):
	def __init__(self, sequences, labels):
		self.sequences = sequences
		self.labels = labels

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		return self.sequences[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
	for epoch in range(epochs):
		model.train()
		for sequences, labels in train_loader:
			sequences, labels = sequences.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(sequences)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
		# Validation step
		model.eval()
		with torch.no_grad():
			val_loss = 0
			for sequences, labels in val_loader:
				sequences, labels = sequences.to(device), labels.to(device)
				outputs = model(sequences)
				loss = criterion(outputs, labels)
				val_loss += loss.item()
		print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}')

def evaluate_model(model, test_loader, criterion):
	model.eval()
	with torch.no_grad():
		test_loss = 0
		for sequences, labels in test_loader:
			sequences, labels = sequences.to(device), labels.to(device)
			outputs = model(sequences)
			loss = criterion(outputs, labels)
			test_loss += loss.item()
	print(f'Test Loss: {test_loss/len(test_loader)}')

def run_model(lower_bound, upper_bound, dataset, **kwargs):
	pooling_size = kwargs['pooling_size']
	Xtrain, Ytrain, Train_mask_label, Xtest, Ytest, Test_mask_label, Xval, Yval, Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(kwargs['left'], kwargs['right'], dataset, padmod=kwargs['padmod'], pooling_size=pooling_size)
	max_len = kwargs['left'] + kwargs['right']

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = MultiheadAttentionModel(max_len, encoding_vectors, 
								 kwargs['save_path'], 
								 kwargs['in_channels'], 
								 kwargs['drop_rate_cnn'], 
								 kwargs['drop_rate_fc'], 
								 kwargs['pooling_size'], 
								 kwargs['pooling_size'], 
								 kwargs['gelu_used'], 
								 kwargs['bn_used'], 
								 kwargs['nb_classes'],
								 kwargs['hidden'], 
								 kwargs['da'], 
								 kwargs['r'],
								 kwargs['return_attention'], 
								 kwargs['attention_regularizer_weight'],
								 kwargs['normalize'], 
								 kwargs['attmod'], 
								 kwargs['sharp_beta'], **kwargs).to(device)
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])

	for i in range(1):
		train_dataset = GeneDataset(Xtrain[i], Ytrain[i])
		val_dataset = GeneDataset(Xval[i], Yval[i])
		test_dataset = GeneDataset(Xtest[i], Ytest[i])

		train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=kwargs['batch_size'], shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=kwargs['batch_size'], shuffle=False)

		train_model(model, train_loader, val_loader, criterion, optimizer, kwargs['epochs'])
		evaluate_model(model, test_loader, criterion)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	'''Model parameters'''

	parser.add_argument('--left', type=int, default=4000, help='set left on sample sequence length')
	parser.add_argument('--right', type=int, default=4000, help='set left on sample sequence length')

	parser.add_argument('--drop_rate_cnn', type=float, default=0.1, help='dropout ratio')
	parser.add_argument('--drop_rate_fc', type=float, default=0.1, help='dropout ratio')
	parser.add_argument('--in_channels', type=int, default=64, help='number of CNN filters') 
	parser.add_argument('--pooling_size', type=int, default=8, help='pooling_size') 
	parser.add_argument('--pooling_stride', type=int, default=8, help='pooling_stride') 
	parser.add_argument('--gelu_used', type=bool, default=True, help='use GELU activation function')
	parser.add_argument('--bn_used', type=bool, default=True, help='use batch normalization')
	parser.add_argument('--nb_classes', type=int, default=6, help='number of classes') # 7 (edit)
	parser.add_argument('--hidden', type=int, default=32, help='number of hidden units of input')
	parser.add_argument('--da', type=int, default=32, help='number of units in attention layer')
	parser.add_argument('--r', type=int, default=8, help='number of heads')
	parser.add_argument('--return_attention', type=bool, default=False, help='return attention of CNN')
	parser.add_argument('--attention_regularizer_weight', type=float, default=0.001, help='attention regularizer weight for attention head')
	parser.add_argument('--normalize', type=bool, default=True, help='normalize attention score')
	parser.add_argument('--attmod', type=str, default='smooth', help='attmod')
	parser.add_argument('--sharp_beta', type=int, default=1, help='sharp Beta')

	parser.add_argument('--dataset', type=str, default='../../mRNAsubloci_train.fasta', help='input sequence data')
	parser.add_argument('--epochs', type=int, default=50, help='')
	parser.add_argument("--lr",type=float,default=0.001,help = 'lr')
	parser.add_argument('--foldnum', type=int, default=8, help='number of cross-validation folds')
	parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
	parser.add_argument("--padmod", type=str,default='after',help="padmod: center, after")

	parser.add_argument('--save_path', type=str, default='model.pth', help='path to save the model')
	parser.add_argument("--message", type=str, default="", help="append to the dir name")

	args = parser.parse_args()
	OUTPATH = os.path.join(basedir,'Results/'+args.message + '/')
	if not os.path.exists(OUTPATH):
		os.makedirs(OUTPATH)
	print('OUTPATH:', OUTPATH)
	del args.message
	
	args.weights_dir = os.path.join(basedir, args.weights_dir)
	
	for k, v in vars(args).items():
		print(k, ':', v)
	
	run_model(**vars(args))



#use the remove data direct from fold
#python3 Multihead_train.py --normalizeatt --classweight --dataset ../direct_8_fold_data/modified_multi_complete_to_cdhit.fasta --epochs 500 --message direct_8fold_model --weights_dir 'model_after_cdhit'


#python3 Multihead_train.py --normalizeatt --classweight --dataset ../modified_multi_complete_to_cdhit.fasta --epochs 500 --message cnn64_smooth_l1
