"""
Custom dataset function for PyTorch data loader
"""
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, x_train, y_train, tau_train):
        # Initialization
        self.x_train = x_train
        self.y_train = y_train
        self.tau_train = tau_train
    
    def __len__(self):
        # Denotes the total number of samples
        return len(self.x_train)
    
    def __getitem__(self, idx):
        # Generates one sample of data
        x = self.x_train[idx]
        y = self.y_train[idx]
        tau = self.tau_train[idx]
        return x, y, tau
