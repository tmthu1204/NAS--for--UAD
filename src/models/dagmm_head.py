import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAGMMHead(nn.Module):
    """
    DAGMM-style density estimator on top of learned latent features.
    - autoencoder compresses and reconstructs source-normal features
    - estimation net predicts mixture assignments from latent + reconstruction cues
    - anomaly score is the mixture energy of the augmented latent descriptor
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        est_hidden_dim: int = 64,
        num_components: int = 3,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.est_hidden_dim = int(est_hidden_dim)
        self.num_components = max(1, int(num_components))
        self.aug_dim = self.latent_dim + 2

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.in_dim),
        )
        self.estimation = nn.Sequential(
            nn.Linear(self.aug_dim, self.est_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.est_hidden_dim, self.num_components),
        )

        self.register_buffer("phi", torch.full((self.num_components,), 1.0 / self.num_components))
        self.register_buffer("mu", torch.zeros(self.num_components, self.aug_dim))
        self.register_buffer("var", torch.ones(self.num_components, self.aug_dim))

    def encode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.encoder(Z)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def reconstruct(self, Z: torch.Tensor):
        latent = self.encode(Z)
        recon = self.decode(latent)
        return latent, recon

    def _reconstruction_features(self, Z: torch.Tensor, recon: torch.Tensor):
        diff = recon - Z
        rel_euc = diff.norm(dim=1, keepdim=True) / Z.norm(dim=1, keepdim=True).clamp_min(1e-6)
        cosine = F.cosine_similarity(Z, recon, dim=1, eps=1e-6).unsqueeze(1)
        return rel_euc, cosine

    def build_augmented_latent(self, Z: torch.Tensor):
        latent, recon = self.reconstruct(Z)
        rel_euc, cosine = self._reconstruction_features(Z, recon)
        augmented = torch.cat([latent, rel_euc, cosine], dim=1)
        recon_mse = ((recon - Z) ** 2).mean(dim=1)
        return augmented, latent, recon, recon_mse

    def estimate_gamma(self, augmented: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.estimation(augmented), dim=1)

    def compute_gmm_params(self, augmented: torch.Tensor, gamma: torch.Tensor):
        gamma_sum = gamma.sum(dim=0).clamp_min(1e-8)
        phi = gamma_sum / gamma_sum.sum()
        mu = torch.matmul(gamma.t(), augmented) / gamma_sum.unsqueeze(1)
        diff = augmented.unsqueeze(1) - mu.unsqueeze(0)
        var = (gamma.unsqueeze(2) * diff.pow(2)).sum(dim=0) / gamma_sum.unsqueeze(1)
        var = var.clamp_min(1e-6)
        return phi, mu, var

    def energy_from_params(self, augmented: torch.Tensor, phi: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
        diff = augmented.unsqueeze(1) - mu.unsqueeze(0)
        log_phi = torch.log(phi.clamp_min(1e-12)).unsqueeze(0)
        log_det = torch.log(var).sum(dim=1).unsqueeze(0)
        quad = (diff.pow(2) / var.unsqueeze(0)).sum(dim=2)
        log_prob = log_phi - 0.5 * (self.aug_dim * math.log(2.0 * math.pi) + log_det + quad)
        return -torch.logsumexp(log_prob, dim=1)

    def covariance_regularizer(self, var: torch.Tensor) -> torch.Tensor:
        return (1.0 / var.clamp_min(1e-6)).sum(dim=1).mean()

    @torch.no_grad()
    def set_mixture_params(self, phi: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
        self.phi.copy_(phi.to(dtype=self.phi.dtype, device=self.phi.device))
        self.mu.copy_(mu.to(dtype=self.mu.dtype, device=self.mu.device))
        self.var.copy_(var.to(dtype=self.var.dtype, device=self.var.device))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        augmented, _, _, _ = self.build_augmented_latent(Z)
        return self.energy_from_params(augmented, self.phi, self.mu, self.var)
