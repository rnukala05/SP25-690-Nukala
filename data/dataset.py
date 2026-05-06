import torch
from torch.utils.data import Dataset
import numpy as np

class ConfusionDataset(Dataset):
    def __init__(self, size=600):
        self.size = size
        self.data = []
        self.labels = []

        for i in range(size):
            seq = np.random.randn(8, 3, 32, 32)
            label = np.random.randint(0, 2)
            self.data.append(seq.astype(np.float32))
            self.labels.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])
