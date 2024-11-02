import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

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
    
class MultiRNAClassifier_CNN(nn.Module):
    def __init__(self, num_classes, num_channels, conv_layers, kernel_size, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        self.layers.append(nn.BatchNorm1d(num_channels))
        self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(kernel_size=2))

        for i in range(1, conv_layers):    
            self.layers.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels * 2, kernel_size=kernel_size))
            self.layers.append(nn.BatchNorm1d(num_channels * 2))
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool1d(kernel_size=2))
            num_channels = num_channels * 2
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(num_channels * 2 * (conv_layers) // 2)
        self.fc = nn.Linear(num_channels * 2 * (conv_layers) // 2, num_classes)

    def forward(self, x):
        feature_maps = []
        
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):  # Capture feature maps after convolutional layers
                feature_maps.append(x)

        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.global_avg_pool(x)
        #drop out 2
        x = self.fc(x)
        
        return x, feature_maps 
       