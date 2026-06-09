import torch
import torch.nn as nn


class MahalanobisHead(nn.Module):
    """
    Learned feature projection followed by a Gaussian descriptor.
    - train-time compactness: squared distance to a learned center
    - test-time score: Mahalanobis distance in the projected space
    """

    def __init__(self, in_dim, hidden_dim=128, rep_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.rep_dim = int(rep_dim)
        self.register_buffer("train_center", torch.zeros(rep_dim))
        self.register_buffer("gauss_mean", torch.zeros(rep_dim))
        self.register_buffer("precision", torch.eye(rep_dim))
        self.register_buffer("covariance", torch.eye(rep_dim))

    def encode(self, Z):
        return self.net(Z)

    @torch.no_grad()
    def init_center(self, Z, eps=1e-3):
        self.eval()
        out = self.encode(Z)
        c = out.mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        self.train_center.copy_(c)

    def compactness_loss(self, Z):
        h = self.encode(Z)
        dist2 = ((h - self.train_center) ** 2).sum(dim=1)
        return dist2.mean(), dist2

    @torch.no_grad()
    def fit_gaussian(self, Z, shrinkage=1e-2):
        h = self.encode(Z)
        mean = h.mean(dim=0)
        centered = h - mean
        n = max(1, int(h.shape[0] - 1))
        cov = centered.T @ centered / float(n)

        shrinkage = float(min(max(shrinkage, 0.0), 1.0))
        eye = torch.eye(self.rep_dim, device=cov.device, dtype=cov.dtype)
        trace_scale = cov.diag().mean()
        if not torch.isfinite(trace_scale) or float(trace_scale.item()) <= 0.0:
            trace_scale = cov.new_tensor(1.0)
        cov = (1.0 - shrinkage) * cov + shrinkage * trace_scale * eye
        cov = cov + 1e-6 * eye
        precision = torch.linalg.inv(cov)

        self.gauss_mean.copy_(mean)
        self.covariance.copy_(cov)
        self.precision.copy_(precision)

    def forward(self, Z):
        h = self.encode(Z)
        delta = h - self.gauss_mean
        mahal = torch.einsum("bi,ij,bj->b", delta, self.precision, delta)
        return torch.clamp(mahal, min=0.0)
