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
        # Convert (960, 640) -> (30 ,640)
        mean_idx = [i for i in range(0, self.embeddings.shape[1], 32)]
        temp = []
        for i in range(len(mean_idx)):
            temp.append(np.mean(self.embeddings[idx][mean_idx[i]: mean_idx[i]+32], axis=0))
        # Use the mean of the RNA-FM embedding across 32 items
        embedding = np.array(temp)
        label = self.labels[idx]
        
        return embedding, label

        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # This layer is to match dimensions if in_channels != out_channels
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride) if in_channels != out_channels else None

    def forward(self, x):
        identity = x  # Store the input for the skip connection
        # Forward pass through the convolutional layers
        out = self.conv1(x)
        out = self.bn1(out)

        # If there's a shortcut path, apply it
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        # Add the shortcut to the output
        out += identity
        out = self.relu(out) 
        return out

class DenseBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(DenseBlock, self).__init__()
        
        self.dense = nn.Linear(input_dim, hidden_dim)  
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)   
        self.dropout = nn.Dropout(dropout_rate)          
        # self.relu = nn.ReLU()                         
                    
    def forward(self, x):
        x = self.dense(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        # x = self.relu(x)
        return x
        

class RNAClassifier_1(nn.Module):
    def __init__(self, num_classes, num_channels, kernel_size, dropout_rate, padding):
        super().__init__()

        self.layer1 = ResidualBlock(in_channels=30, out_channels=num_channels, kernel_size=kernel_size, padding=padding)
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=30, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(30)
        self.relu = nn.ReLU()

        # Adaptive pooling directly after the last residual block
        self.adaptive_pool = nn.AdaptiveAvgPool1d(512)
        # Define Dense Blocks
        self.dense1 = DenseBlock(input_dim=512, hidden_dim=96, dropout_rate=dropout_rate)  # Adjust input_dim based on adaptive pooling
        # self.dense2 = DenseBlock(input_dim=256, hidden_dim=96, dropout_rate=dropout_rate)
        self.dense2 = DenseBlock(input_dim=96, hidden_dim=num_classes, dropout_rate=dropout_rate)  # Output should match num_classes

    def forward(self, x):
        # Forward through residual blocks
        x = self.layer1(x)
        feature_map = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape) # Shape: [batch_size, 30, 640]
        x = self.adaptive_pool(x)  # Shape: [batch_size, 30, 512]

        # # Reshape for the fully connected layers
        batch_size, num_segments, _ = x.size()  # num_channels should be 30, output from adaptive pooling
        x = x.view(batch_size * num_segments, -1)   # Flatten to [batch_size, 30 * 512]
        x = self.dense1(x)  # First dense block
        # x = self.dense2(x)  # Second dense block
        x = self.dense2(x)  # Third dense block
        x = x.view(batch_size, num_segments, -1)

        return x, feature_map  # Return final output and feature maps (if needed)


class RNAClassifier_2(nn.Module):
    def __init__(self, num_classes, num_channels, kernel_size, dropout_rate, padding):
        super().__init__()

        self.layer1 = ResidualBlock(in_channels=30, out_channels=num_channels, kernel_size=kernel_size, padding=padding)
        self.layer2 = ResidualBlock(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding)
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=30, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(30)
        self.relu = nn.ReLU()

        self.adaptive_pool = nn.AdaptiveAvgPool1d(512)
        self.dense_1 = DenseBlock(input_dim=512, hidden_dim=256, dropout_rate=dropout_rate)  # Adjust input_dim based on adaptive pooling
        self.dense_2 = DenseBlock(input_dim=256, hidden_dim=64, dropout_rate=dropout_rate)
        self.dense_3 = DenseBlock(input_dim=64, hidden_dim=num_classes, dropout_rate=dropout_rate)  # Output should match num_classes
        self.bn_dense = nn.BatchNorm1d(num_classes)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        feature_map = x
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        # print(x.shape) # Shape: [batch_size, 30, 640]
        x = self.adaptive_pool(x)  # Shape: [batch_size, 30, 512]

        # Reshape for the fully connected layers
        batch_size, num_segments, _ = x.size()  # num_channels should be 30, output from adaptive pooling
        x = x.view(batch_size * num_segments, -1)   # Flatten to [batch_size, 30 * 512]
        x = self.dense_1(x) 
        x = self.dense_2(x)
        x = self.dense_3(x) 
        x = self.bn_dense(x)
        x = x.view(batch_size, num_segments, -1)
        x = self.sigm(x)

        return x, feature_map  # Return final output and feature maps (if needed)