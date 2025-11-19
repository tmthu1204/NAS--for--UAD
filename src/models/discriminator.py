import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))


    def forward(self, x):
        return self.net(x)