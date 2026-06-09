import math

import torch
import torch.nn as nn


class GMMHead(nn.Module):
    """
    Learned feature projection followed by a Gaussian Mixture descriptor.
    - train-time warmup: squared distance to a learned center
    - train-time density fitting: negative log-likelihood under a fitted GMM
    - test-time score: negative log-likelihood in the projected space
    """

    def __init__(self, in_dim, hidden_dim=128, rep_dim=64, num_components=3, covariance_type="diag"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.rep_dim = int(rep_dim)
        self.num_components = max(1, int(num_components))
        self.covariance_type = str(covariance_type).strip().lower()
        if self.covariance_type not in {"diag", "full"}:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")

        self.register_buffer("train_center", torch.zeros(rep_dim))
        self.register_buffer("weights", torch.full((self.num_components,), 1.0 / self.num_components))
        self.register_buffer("means", torch.zeros(self.num_components, rep_dim))
        if self.covariance_type == "diag":
            self.register_buffer("precisions_cholesky", torch.ones(self.num_components, rep_dim))
        else:
            eye = torch.eye(rep_dim).unsqueeze(0).repeat(self.num_components, 1, 1)
            self.register_buffer("precisions_cholesky", eye)

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
    def set_gmm_params(self, weights, means, precisions_cholesky):
        weights_t = torch.as_tensor(weights, dtype=self.weights.dtype, device=self.weights.device)
        means_t = torch.as_tensor(means, dtype=self.means.dtype, device=self.means.device)
        precisions_t = torch.as_tensor(
            precisions_cholesky,
            dtype=self.precisions_cholesky.dtype,
            device=self.precisions_cholesky.device,
        )

        if weights_t.shape != self.weights.shape:
            raise ValueError(f"Unexpected weights shape: {tuple(weights_t.shape)} vs {tuple(self.weights.shape)}")
        if means_t.shape != self.means.shape:
            raise ValueError(f"Unexpected means shape: {tuple(means_t.shape)} vs {tuple(self.means.shape)}")
        if precisions_t.shape != self.precisions_cholesky.shape:
            raise ValueError(
                "Unexpected precisions_cholesky shape: "
                f"{tuple(precisions_t.shape)} vs {tuple(self.precisions_cholesky.shape)}"
            )

        self.weights.copy_(weights_t)
        self.means.copy_(means_t)
        self.precisions_cholesky.copy_(precisions_t)

    def _estimate_log_gaussian_prob(self, H):
        n_features = int(H.shape[1])
        log_2pi = math.log(2.0 * math.pi)

        if self.covariance_type == "diag":
            precisions = self.precisions_cholesky
            log_det = torch.log(precisions).sum(dim=1)
            diff = H.unsqueeze(1) - self.means.unsqueeze(0)
            y = diff * precisions.unsqueeze(0)
            quad = (y * y).sum(dim=2)
            return log_det.unsqueeze(0) - 0.5 * (n_features * log_2pi + quad)

        diff = H.unsqueeze(1) - self.means.unsqueeze(0)
        y = torch.einsum("bki,kij->bkj", diff, self.precisions_cholesky)
        quad = (y * y).sum(dim=2)
        diag = torch.diagonal(self.precisions_cholesky, dim1=1, dim2=2)
        log_det = torch.log(diag).sum(dim=1)
        return log_det.unsqueeze(0) - 0.5 * (n_features * log_2pi + quad)

    def negative_log_likelihood_from_latent(self, H):
        log_prob = self._estimate_log_gaussian_prob(H)
        log_weights = torch.log(torch.clamp(self.weights, min=1e-12)).unsqueeze(0)
        log_mix = torch.logsumexp(log_prob + log_weights, dim=1)
        return -log_mix

    def negative_log_likelihood(self, Z):
        h = self.encode(Z)
        return self.negative_log_likelihood_from_latent(h)

    def forward(self, Z):
        return self.negative_log_likelihood(Z)
