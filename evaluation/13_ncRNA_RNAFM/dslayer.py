import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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