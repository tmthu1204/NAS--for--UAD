import torch
import torch.nn as nn


class LOFHead(nn.Module):
    """
    Learned feature projection followed by a Local Outlier Factor descriptor.
    - train-time compactness: squared distance to a learned center
    - test-time score: LOF-based local density anomaly score in projected space
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, rep_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.rep_dim = int(rep_dim)
        self.register_buffer("train_center", torch.zeros(rep_dim))

    def encode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.net(Z)

    @torch.no_grad()
    def init_center(self, Z: torch.Tensor, eps: float = 1e-3):
        self.eval()
        out = self.encode(Z)
        c = out.mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        self.train_center.copy_(c)

    def compactness_loss(self, Z: torch.Tensor):
        h = self.encode(Z)
        dist2 = ((h - self.train_center) ** 2).sum(dim=1)
        return dist2.mean(), dist2
