from __future__ import annotations

import numpy as np
import torch

from src.models.mahalanobis_head import MahalanobisHead

from .base import OneClassBackend, OneClassConfig


class MahalanobisHeadBackend(OneClassBackend):
    method_name = "mahalanobis_head"
    score_name = "mahalanobis_dist2"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None

    def _build_model(self, device: str):
        self.model = MahalanobisHead(
            in_dim=self.in_dim,
            hidden_dim=max(1, int(self.config.maha_hidden_dim)),
            rep_dim=max(1, int(self.config.maha_rep_dim)),
        ).to(device)
        return self.model

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        Zs = torch.tensor(np.asarray(Zs_np), dtype=torch.float32, device=device)
        model = self._build_model(device)
        model.init_center(Zs)

        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        n = Zs.shape[0]

        for _ in range(self.config.epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.config.batch_size):
                zb = Zs[perm[i:i + self.config.batch_size]]
                loss, _ = model.compactness_loss(zb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        model.fit_gaussian(Zs, shrinkage=float(self.config.maha_shrinkage))
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("MahalanobisHeadBackend must be fit before scoring.")
        return self.model(Z)

    def summary(self) -> dict:
        out = super().summary()
        if self.model is None:
            out["rep_dim"] = None
            out["cov_trace"] = None
        else:
            out["rep_dim"] = int(self.model.rep_dim)
            out["cov_trace"] = float(torch.trace(self.model.covariance).detach().cpu().item())
        return out
