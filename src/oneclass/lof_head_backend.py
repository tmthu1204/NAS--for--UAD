from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor

from src.models.lof_head import LOFHead

from .base import OneClassBackend, OneClassConfig


class LOFHeadBackend(OneClassBackend):
    method_name = "lof_head"
    score_name = "lof_neg_score_samples"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None
        self.lof = None
        self.effective_n_neighbors = None

    def _build_model(self, device: str):
        self.model = LOFHead(
            in_dim=self.in_dim,
            hidden_dim=max(1, int(self.config.lof_hidden_dim)),
            rep_dim=max(1, int(self.config.lof_rep_dim)),
        ).to(device)
        return self.model

    def _fit_lof(self, H_np: np.ndarray):
        n_samples = int(H_np.shape[0])
        n_neighbors = max(1, min(int(self.config.lof_n_neighbors), max(1, n_samples - 1)))
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            metric=str(self.config.lof_metric).strip().lower(),
            p=int(self.config.lof_p),
            novelty=True,
            n_jobs=1,
        )
        lof.fit(H_np)
        self.effective_n_neighbors = int(n_neighbors)
        return lof

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
        H_np = model.encode(Zs).detach().cpu().numpy().astype(np.float32, copy=False)
        self.lof = self._fit_lof(H_np)
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None or self.lof is None:
            raise RuntimeError("LOFHeadBackend must be fit before scoring.")
        H_np = self.model.encode(Z).detach().cpu().numpy().astype(np.float32, copy=False)
        scores = -self.lof.score_samples(H_np)
        return torch.as_tensor(scores, dtype=Z.dtype, device=Z.device)

    def summary(self) -> dict:
        out = super().summary()
        out["rep_dim"] = None if self.model is None else int(self.model.rep_dim)
        out["n_neighbors"] = self.effective_n_neighbors
        out["metric"] = str(self.config.lof_metric).strip().lower()
        out["p"] = int(self.config.lof_p)
        return out
