"""Dataset utilities and augmentations."""
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        self.w = None if w is None else torch.tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is None and self.w is None:
            return x
        if self.y is None and self.w is not None:
            return x, self.w[idx]          # ✅ unlabeled + weight
        if self.y is not None and self.w is None:
            return x, self.y[idx]
        return x, self.y[idx], self.w[idx]









# Augmentations
class Jitter:
    def __init__(self, sigma=0.01):
        self.sigma = sigma


    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma




class Scale:
    def __init__(self, low=0.8, high=1.2):
        self.low, self.high = low, high


    def __call__(self, x):
        factor = torch.empty(x.size(0),1,1).uniform_(self.low, self.high)
        return x * factor




class Permutation:
    def __init__(self, M=5):
        self.M = M


    def __call__(self, x):
        B, T, C = x.shape
        seg = T // self.M
        x = x[:, :seg*self.M, :].reshape(B, self.M, seg, C)
        out = []
        for i in range(B):
            idx = torch.randperm(self.M)
            out.append(x[i, idx].reshape(seg*self.M, C))
        return torch.stack(out)