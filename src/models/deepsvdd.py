import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSVDD(nn.Module):
    """
    Deep SVDD Soft-boundary.
    - input: feature Z shape [B, D]
    - output: dist2 to center c
    """
    def __init__(self, in_dim, hidden_dim=128, rep_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.rep_dim = rep_dim
        self.register_buffer("c", torch.zeros(rep_dim))
        self.R = nn.Parameter(torch.tensor(0.0))

    @torch.no_grad()
    def init_center(self, Z, eps=1e-3):
        self.eval()
        out = self.net(Z)
        c = out.mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        self.c.copy_(c)

    def forward(self, Z):
        h = self.net(Z)
        dist2 = ((h - self.c) ** 2).sum(dim=1)
        return dist2

    def loss_soft_boundary(self, Z, nu=0.1):
        dist2 = self.forward(Z)
        R2 = self.R ** 2
        slack = F.relu(dist2 - R2)
        loss = R2 + (1.0 / nu) * slack.mean()
        return loss, dist2, slack
