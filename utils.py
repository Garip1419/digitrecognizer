from torch.utils.data import Dataset, DataLoader, random_split, default_collate
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

#Ref: https://machinelearningmastery.com/training-a-pytorch-model-with-dataloader-and-dataset/
class digitdataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
 
    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target