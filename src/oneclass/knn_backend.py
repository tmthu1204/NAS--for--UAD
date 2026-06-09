from __future__ import annotations

import numpy as np
import torch

from .base import OneClassBackend, OneClassConfig


class KNNDistanceBackend(OneClassBackend):
    method_name = "knn_distance"
    score_name = "knn_distance"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.reference = None
        self.k = max(1, int(config.knn_k))

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.reference = torch.tensor(np.asarray(Zs_np), dtype=torch.float32, device=device)
        self.k = max(1, min(self.k, int(self.reference.shape[0])))
        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.reference is None:
            raise RuntimeError("KNNDistanceBackend must be fit before scoring.")

        dists = torch.cdist(Z, self.reference, p=2)
        topk = torch.topk(dists, k=self.k, dim=1, largest=False).values
        return topk[:, -1]

    def summary(self) -> dict:
        out = super().summary()
        out["k"] = int(self.k)
        out["reference_count"] = 0 if self.reference is None else int(self.reference.shape[0])
        return out
