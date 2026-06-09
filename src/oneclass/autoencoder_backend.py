from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import OneClassBackend, OneClassConfig


class FeatureAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(Z)
        return self.decoder(latent)


class AutoEncoderBackend(OneClassBackend):
    method_name = "autoencoder"
    score_name = "reconstruction_mse"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None

    def _build_model(self, device: str):
        hidden_dim = max(1, int(self.config.ae_hidden_dim))
        latent_dim = max(1, int(self.config.ae_latent_dim))
        self.model = FeatureAutoEncoder(
            in_dim=self.in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
        return self.model

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        Zs = torch.tensor(np.asarray(Zs_np), dtype=torch.float32, device=device)

        model = self._build_model(device)
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        n = Zs.shape[0]

        for _ in range(self.config.epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.config.batch_size):
                zb = Zs[perm[i:i + self.config.batch_size]]
                recon = model(zb)
                loss = ((recon - zb) ** 2).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("AutoEncoderBackend must be fit before scoring.")
        recon = self.model(Z)
        return ((recon - Z) ** 2).mean(dim=1)
