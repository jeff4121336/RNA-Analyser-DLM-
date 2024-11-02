import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.functional as F

class RNATypeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings  # Expecting shape (num_samples, L, embedding_dim)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Use the mean of the RNA-FM embedding along the sequence dimension
        # Convert (L, 640) -> (640,)
        embedding = np.mean(self.embeddings[idx], axis=0)
        label = self.labels[idx]
        
        return embedding, label
    
class RNATypeClassifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.fc = nn.Linear(640, num_class)

    def forward(self, x):
        x = self.fc(x)

        return x
    
class MultiRNALabelClassifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=640, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_class)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embedding_size)
        
        # Transpose to (batch_size, embedding_size, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 640, seq_length)
        
        # Convolutional layers, conv layer + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # Output Shape: (batch_size, 64, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)

        # Fully connected layer for multi-label classification
        x = self.fc(x)  # Output Shape: (batch_size, num_classes)
        
        return torch.sigmoid(x)  # Sigmoid for multi-label output (probability)