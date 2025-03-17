import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from hier_attention_mask import AttentionMask
from sklearn.metrics import (
	roc_auc_score,
	average_precision_score,
	matthews_corrcoef,
	roc_curve,
	precision_recall_curve,
)

# class PostitionEmbedding(nn.Module):


class MultiscaleCNN(nn.Module):
	def __init__(
		self,
		max_len,
		embedding_vec,
		save_path,
		in_channels=64,
		drop_rate_cnn=0.2,
		drop_rate_fc=0.2,
		pooling_size=8,
		pooling_stride=8,
		gelu_used=True,
		bn_used=True,
		nb_classes=6,
		hidden=32,
		da=32,
		r=8,
		returnAttention=False,
		attentionRegularizerWeight=0.0,
		normalize=True,
		attmod="smooth",
		sharpBeta=1
	):
		super(MultiscaleCNN, self).__init__()
		self.max_len = max_len
		self.embedding_vec = embedding_vec
		self.in_channels = in_channels
		self.drop_rate_cnn = drop_rate_cnn
		self.drop_rate_fc = drop_rate_fc
		self.pooling_size = pooling_size
		self.pooling_stride = pooling_stride
		self.gelu_used = gelu_used
		self.bn_used = bn_used,
		self.nb_classes = nb_classes
		self.save_path = save_path
		self.hidden = hidden
		self.da = da
		self.r = r
		self.returnAttention = returnAttention
		self.attentionRegularizerWeight = attentionRegularizerWeight
		self.normalize = normalize
		self.attmod = attmod
		self.sharpBeta = sharpBeta

		# Embedding Mechanism here # Change to one-hot encoding
		# self.embedding = nn.Embedding(
		#     num_embeddings=1000, embedding_dim=128
		# )
		
		self.bn1 = nn.BatchNorm1d(in_channels)
		self.bn2 = nn.BatchNorm1d(in_channels // 2)

		# CNN1
		self.conv1_1 = nn.Conv1d(
			in_channels=embedding_vec.shape[1],
			out_channels=in_channels,
			kernel_size=9,
			padding="same",
		)
		self.conv1_2 = nn.Conv1d(
			in_channels=in_channels,
			out_channels=in_channels // 2,
			kernel_size=9,
			padding="same",
		)

		# CNN2
		self.conv2_1 = nn.Conv1d(
			in_channels=embedding_vec.shape[1],
			out_channels=in_channels,
			kernel_size=20,
			padding="same",
		)
		self.conv2_2 = nn.Conv1d(
			in_channels=in_channels,
			out_channels=in_channels // 2,
			kernel_size=20,
			padding="same",
		)

		# CNN3
		self.conv3_1 = nn.Conv1d(
			in_channels=embedding_vec.shape[1],
			out_channels=in_channels // 2,
			kernel_size=49,
			padding="same",
		)
		self.conv3_2 = nn.Conv1d(
			in_channels=in_channels // 2,
			out_channels=in_channels // 2,
			kernel_size=49,
			padding="same",
		)

		# Maxpooling layers
		self.pool = nn.MaxPool1d(
			kernel_size=self.pooling_size, stride=self.pooling_stride
		)

		# Head for Multihead Attention
		self.attentionHead = AttentionMask(
			hidden=self.hidden,
			da=self.da,
			r=self.r,
			returnAttention=self.returnAttention,
			attentionRegularizerWeight=self.attentionRegularizerWeight,
			normalize=self.normalize,
			attmod=self.attmod,
			sharpBeta=self.sharpBeta
		)

		# Activation functions
		if self.gelu_used:
			self.activation = nn.GELU()
		else:
			self.activation = nn.ReLU()

		# Dropout layers
		self.dropout_cnn = nn.Dropout(self.drop_rate_cnn)
		self.dropout_fc = nn.Dropout(self.drop_rate_fc)

		# Flatten layer (2D)
		self.flatten = nn.Flatten()

		# Fully connected layer (Edit needed)
		self.fc = nn.Linear(100, nb_classes)

	def CNN(self, x, conv1, conv2, bn1, bn2):
		x = conv1(x)
		if self.bn_used:
			x = self.activation(bn1(x))
		else:
			x = self.activation(x)
		x = conv2(x)
		if self.bn_used:
			x = self.activation(bn2(x))
		else:
			x = self.activation(x)
		x = self.pool(x)
		x, regularization_loss, attention = self.attentionHead(x)
		x = self.dropout_cnn(x)
		return x, regularization_loss, attention

	def forward(self, x, mask):
		# One-hot encoding (modify needed)
		x = F.one_hot(x, num_classes=4).float()
		x = self.dropout_input(x)
		x = x.permute(0, 2, 1)  # Change to (batch_size, channels, seq_len)

		# CNN1, CNN2, CNN3
		x1, x1_regularization_loss, x1_attention = self.CNN(x, self.conv1_1, self.conv1_2, self.bn1, self.bn2)
		x2, x2_regularization_loss, x2_attention = self.CNN(x, self.conv2_1, self.conv2_2, self.bn1, self.bn2)
		x3, x3_regularization_loss, x3_attention = self.CNN(x, self.conv3_1, self.conv3_2, self.bn1, self.bn2)

		# Concatenate and flatten
		x = torch.cat((x1, x2, x3), dim=1) # (batch_size, combined_channels, seq_len)
		x = self.flatten(x) # (batch_size, combined_channels * seq_len)
		x = self.dropout_fc(x)
		x = self.fc(x) # (batch_size, nb_classes)

		return x

# Example usage:
# model = MultiheadAttentionModel(max_len=100, nb_classes=10, save_path='/path/to/save', kfold_index=1)
# output = model(torch.randint(0, 1000, (32, 100)), None)
