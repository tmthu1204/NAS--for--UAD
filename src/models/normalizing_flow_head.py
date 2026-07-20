import math

import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, scale_clip: float = 2.0):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.scale_clip = float(scale_clip)
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.dim),
        )
        self.register_buffer("mask", mask.reshape(1, self.dim).float())

    def _scale_shift(self, x_masked: torch.Tensor):
        st = self.net(x_masked)
        s_raw, t = st.chunk(2, dim=1)
        s = torch.tanh(s_raw) * self.scale_clip
        return s, t

    def forward(self, x: torch.Tensor):
        x_masked = x * self.mask
        s, t = self._scale_shift(x_masked)
        inv_mask = 1.0 - self.mask
        y = x_masked + inv_mask * (x * torch.exp(s) + t)
        log_det = (inv_mask * s).sum(dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        y_masked = y * self.mask
        s, t = self._scale_shift(y_masked)
        inv_mask = 1.0 - self.mask
        x = y_masked + inv_mask * ((y - t) * torch.exp(-s))
        log_det = -(inv_mask * s).sum(dim=1)
        return x, log_det


class NormalizingFlowHead(nn.Module):
    """
    Learned feature projection followed by an invertible affine-coupling flow.
    - train-time warmup: squared distance to a learned center
    - train-time density fitting: negative log-likelihood under the flow
    - test-time score: negative log-likelihood in projected latent space
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, rep_dim: int = 64, flow_layers: int = 4, scale_clip: float = 2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.rep_dim = int(rep_dim)
        self.flow_layers = max(1, int(flow_layers))
        self.scale_clip = float(scale_clip)
        self.register_buffer("train_center", torch.zeros(self.rep_dim))

        base_mask = (torch.arange(self.rep_dim) % 2).float()
        flows = []
        for i in range(self.flow_layers):
            mask = base_mask if i % 2 == 0 else 1.0 - base_mask
            flows.append(
                AffineCoupling(
                    dim=self.rep_dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    scale_clip=self.scale_clip,
                )
            )
        self.flows = nn.ModuleList(flows)

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

    def negative_log_likelihood_from_latent(self, H: torch.Tensor) -> torch.Tensor:
        z = H
        log_det_sum = torch.zeros(H.shape[0], dtype=H.dtype, device=H.device)
        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z)
            log_det_sum = log_det_sum + log_det

        log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
        log_prob = log_base + log_det_sum
        return -log_prob

    def negative_log_likelihood(self, Z: torch.Tensor) -> torch.Tensor:
        h = self.encode(Z)
        return self.negative_log_likelihood_from_latent(h)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.negative_log_likelihood(Z)
