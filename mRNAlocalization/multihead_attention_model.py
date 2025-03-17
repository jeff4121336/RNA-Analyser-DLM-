import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        self, max_len, in_channels, drop_rate, gelu_used, nb_classes, save_path
    ):
        super(MultiscaleCNN, self).__init__()
        self.max_len = max_len
        self.nb_classes = nb_classes
        self.drop_rate = drop_rate
        self.gelu_used = gelu_used

        self.save_path = save_path

        self.embedding = nn.Embedding(
            num_embeddings=1000, embedding_dim=128
        )  # Adjust num_embeddings and embedding_dim as needed
        self.dropout_input = nn.Dropout(0.2)

        # CNN1
        self.conv1_1 = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels / 2, kernel_size=9
        )
        self.conv1_2 = nn.Conv1d(
            in_channels=in_channels / 2, out_channels=in_channels / 2, kernel_size=9
        )
        # CNN2
        self.conv2_1 = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels / 2, kernel_size=20
        )
        self.conv2_2 = nn.Conv1d(
            in_channels=in_channels / 2, out_channels=in_channels / 2, kernel_size=20
        )
        # CNN3
        self.conv3_1 = nn.Conv1d(
            in_channels=in_channels / 2, out_channels=in_channels / 2, kernel_size=49
        )
        self.conv3_2 = nn.Conv1d(
            in_channels=in_channels / 2, out_channels=in_channels / 2, kernel_size=49
        )

        # Maxpooling layers
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

        # Head for Multihead Attention
        self.attentionHead = AttentionMask()

        # Activation functions
        self.relu = nn.ReLU()
        self.gelu = nn.GeLU()

        # Dropout layers
        self.dropout = nn.Dropout(0.2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(32 * 3, nb_classes)