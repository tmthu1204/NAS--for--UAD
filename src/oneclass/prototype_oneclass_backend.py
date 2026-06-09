from __future__ import annotations

import numpy as np
import torch

from src.models.prototype_oneclass import PrototypeOneClass

from .base import OneClassBackend, OneClassConfig


class PrototypeOneClassBackend(OneClassBackend):
    method_name = "prototype_oneclass"
    score_name = "min_proto_dist2"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None

    def _build_model(self, device: str):
        self.model = PrototypeOneClass(
            in_dim=self.in_dim,
            hidden_dim=max(1, int(self.config.proto_hidden_dim)),
            rep_dim=max(1, int(self.config.proto_rep_dim)),
            num_prototypes=max(1, int(self.config.proto_count)),
        ).to(device)
        return self.model

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        Zs = torch.tensor(np.asarray(Zs_np), dtype=torch.float32, device=device)

        model = self._build_model(device)
        model.init_prototypes(Zs)

        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        n = Zs.shape[0]

        for _ in range(self.config.epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.config.batch_size):
                zb = Zs[perm[i:i + self.config.batch_size]]
                loss, _, _ = model.loss(
                    zb,
                    separation_weight=float(self.config.proto_separation_weight),
                    separation_margin=float(self.config.proto_separation_margin),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("PrototypeOneClassBackend must be fit before scoring.")
        return self.model(Z)

    def summary(self) -> dict:
        out = super().summary()
        out["prototype_count"] = 0 if self.model is None else int(self.model.num_prototypes)
        if self.model is None:
            out["prototype_min_pairwise_dist2"] = None
        elif self.model.num_prototypes <= 1:
            out["prototype_min_pairwise_dist2"] = 0.0
        else:
            with torch.no_grad():
                proto_dist2 = self.model.pairwise_dist2(self.model.prototypes, self.model.prototypes)
                mask = ~torch.eye(
                    self.model.num_prototypes,
                    device=proto_dist2.device,
                    dtype=torch.bool,
                )
                out["prototype_min_pairwise_dist2"] = float(proto_dist2[mask].min().detach().cpu().item())
        return out
