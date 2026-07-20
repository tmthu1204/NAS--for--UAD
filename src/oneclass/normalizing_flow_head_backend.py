from __future__ import annotations

import numpy as np
import torch

from src.models.normalizing_flow_head import NormalizingFlowHead

from .base import OneClassBackend, OneClassConfig


class NormalizingFlowHeadBackend(OneClassBackend):
    method_name = "normalizing_flow_head"
    score_name = "flow_neg_log_likelihood"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None

    def _build_model(self, device: str):
        hidden_dim = max(1, int(self.config.flow_hidden_dim))
        rep_dim = max(2, int(self.config.flow_rep_dim))
        flow_layers = max(1, int(self.config.flow_layers))
        self.model = NormalizingFlowHead(
            in_dim=self.in_dim,
            hidden_dim=hidden_dim,
            rep_dim=rep_dim,
            flow_layers=flow_layers,
            scale_clip=float(self.config.flow_scale_clip),
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
        warmup_epochs = max(1, min(int(self.config.flow_warmup_epochs), int(self.config.epochs)))

        for ep in range(self.config.epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.config.batch_size):
                zb = Zs[perm[i:i + self.config.batch_size]]
                if ep < warmup_epochs:
                    loss, _ = model.compactness_loss(zb)
                else:
                    loss = model.negative_log_likelihood(zb).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("NormalizingFlowHeadBackend must be fit before scoring.")
        return self.model(Z)

    def summary(self) -> dict:
        out = super().summary()
        if self.model is None:
            out["rep_dim"] = None
            out["flow_layers"] = None
            out["scale_clip"] = None
        else:
            out["rep_dim"] = int(self.model.rep_dim)
            out["flow_layers"] = int(self.model.flow_layers)
            out["scale_clip"] = float(self.model.scale_clip)
        return out
