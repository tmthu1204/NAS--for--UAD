from __future__ import annotations

import numpy as np
import torch

from src.models.dagmm_head import DAGMMHead

from .base import OneClassBackend, OneClassConfig


class DAGMMHeadBackend(OneClassBackend):
    method_name = "dagmm_head"
    score_name = "dagmm_energy"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None

    def _build_model(self, device: str, num_components: int):
        self.model = DAGMMHead(
            in_dim=self.in_dim,
            hidden_dim=max(1, int(self.config.dagmm_hidden_dim)),
            latent_dim=max(1, int(self.config.dagmm_latent_dim)),
            est_hidden_dim=max(1, int(self.config.dagmm_est_hidden_dim)),
            num_components=max(1, int(num_components)),
        ).to(device)
        return self.model

    @torch.no_grad()
    def _refresh_mixture_params(self, Zs: torch.Tensor):
        if self.model is None:
            raise RuntimeError("DAGMMHeadBackend model must be built before refreshing mixture params.")
        augmented, _, _, _ = self.model.build_augmented_latent(Zs)
        gamma = self.model.estimate_gamma(augmented)
        phi, mu, var = self.model.compute_gmm_params(augmented, gamma)
        self.model.set_mixture_params(phi, mu, var)

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        Zs = torch.tensor(np.asarray(Zs_np), dtype=torch.float32, device=device)
        effective_components = max(1, min(int(self.config.dagmm_components), int(Zs.shape[0])))
        model = self._build_model(device, num_components=effective_components)

        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        n = Zs.shape[0]
        warmup_epochs = max(1, min(int(self.config.dagmm_warmup_epochs), int(self.config.epochs)))
        lambda_energy = float(max(self.config.dagmm_lambda_energy, 0.0))
        lambda_cov_diag = float(max(self.config.dagmm_lambda_cov_diag, 0.0))

        for ep in range(self.config.epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.config.batch_size):
                zb = Zs[perm[i:i + self.config.batch_size]]
                augmented, _, _, recon_mse = model.build_augmented_latent(zb)
                recon_loss = recon_mse.mean()

                if ep < warmup_epochs:
                    loss = recon_loss
                else:
                    gamma = model.estimate_gamma(augmented)
                    phi, mu, var = model.compute_gmm_params(augmented, gamma)
                    energy = model.energy_from_params(augmented, phi, mu, var).mean()
                    cov_diag_penalty = model.covariance_regularizer(var)
                    loss = recon_loss + lambda_energy * energy + lambda_cov_diag * cov_diag_penalty

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        self._refresh_mixture_params(Zs)
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DAGMMHeadBackend must be fit before scoring.")
        return self.model(Z)

    def summary(self) -> dict:
        out = super().summary()
        if self.model is None:
            out["latent_dim"] = None
            out["est_hidden_dim"] = None
            out["num_components"] = None
        else:
            out["latent_dim"] = int(self.model.latent_dim)
            out["est_hidden_dim"] = int(self.model.est_hidden_dim)
            out["num_components"] = int(self.model.num_components)
        out["lambda_energy"] = float(self.config.dagmm_lambda_energy)
        out["lambda_cov_diag"] = float(self.config.dagmm_lambda_cov_diag)
        out["warmup_epochs"] = int(self.config.dagmm_warmup_epochs)
        return out
