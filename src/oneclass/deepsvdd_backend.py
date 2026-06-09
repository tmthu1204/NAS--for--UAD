from __future__ import annotations

import numpy as np
import torch

from src.models.deepsvdd import DeepSVDD

from .base import OneClassBackend, OneClassConfig


class DeepSVDDBackend(OneClassBackend):
    method_name = "deepsvdd"
    score_name = "dist2"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None

    def _build_model(self, device: str):
        self.model = DeepSVDD(
            in_dim=self.in_dim,
            hidden_dim=self.config.svdd_hidden_dim,
            rep_dim=self.config.svdd_rep_dim,
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

        for ep in range(self.config.epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.config.batch_size):
                zb = Zs[perm[i:i + self.config.batch_size]]
                if ep < self.config.svdd_warmup_epochs:
                    dist2 = model(zb)
                    loss = dist2.mean()
                else:
                    loss, _, _ = model.loss_soft_boundary(zb, nu=self.config.svdd_nu)

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DeepSVDDBackend must be fit before scoring.")
        return self.model(Z)

    def summary(self) -> dict:
        out = super().summary()
        out["radius"] = None if self.model is None else float(self.model.R.detach().cpu().item())
        return out
