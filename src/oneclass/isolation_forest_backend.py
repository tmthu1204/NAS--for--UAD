from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from .base import OneClassBackend, OneClassConfig


class IsolationForestBackend(OneClassBackend):
    method_name = "isolation_forest"
    score_name = "neg_score_samples"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None
        self.effective_max_samples = None

    def _parse_max_samples(self):
        raw = self.config.iforest_max_samples
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value == "auto":
                return "auto"
            if "." in value:
                return float(value)
            return int(value)
        if isinstance(raw, float) and 0.0 < raw <= 1.0:
            return float(raw)
        return int(raw)

    def _parse_contamination(self):
        raw = self.config.iforest_contamination
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value == "auto":
                return "auto"
            return float(value)
        return float(raw)

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        del device
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        Z_fit = np.asarray(Zs_np, dtype=np.float32)
        model = IsolationForest(
            n_estimators=max(1, int(self.config.iforest_n_estimators)),
            max_samples=self._parse_max_samples(),
            contamination=self._parse_contamination(),
            max_features=float(self.config.iforest_max_features),
            random_state=seed,
            n_jobs=1,
        )
        model.fit(Z_fit)

        self.model = model
        self.effective_max_samples = int(model.max_samples_)
        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("IsolationForestBackend must be fit before scoring.")
        Z_np = Z.detach().cpu().numpy().astype(np.float32, copy=False)
        scores = -self.model.score_samples(Z_np)
        return torch.as_tensor(scores, dtype=Z.dtype, device=Z.device)

    def summary(self) -> dict:
        out = super().summary()
        out["n_estimators"] = None if self.model is None else int(len(self.model.estimators_))
        out["max_samples"] = self.effective_max_samples
        out["max_features"] = float(self.config.iforest_max_features)
        out["contamination"] = self.config.iforest_contamination
        return out
