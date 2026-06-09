from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class OneClassConfig:
    method: str = "deepsvdd"
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 1024
    max_fit: int | None = None
    knn_k: int = 5
    ocsvm_nu: float = 0.05
    ocsvm_kernel: str = "rbf"
    ocsvm_gamma: str = "scale"
    ocsvm_degree: int = 3
    ocsvm_coef0: float = 0.0
    svdd_hidden_dim: int = 128
    svdd_rep_dim: int = 64
    svdd_nu: float = 0.05
    svdd_warmup_epochs: int = 2
    ae_hidden_dim: int = 128
    ae_latent_dim: int = 64
    maha_hidden_dim: int = 128
    maha_rep_dim: int = 64
    maha_shrinkage: float = 1e-2
    gmm_hidden_dim: int = 128
    gmm_rep_dim: int = 64
    gmm_components: int = 3
    gmm_covariance_type: str = "diag"
    gmm_reg_covar: float = 1e-4
    gmm_warmup_epochs: int = 2
    proto_hidden_dim: int = 128
    proto_rep_dim: int = 64
    proto_count: int = 4
    proto_separation_weight: float = 0.1
    proto_separation_margin: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class OneClassBackend(nn.Module, ABC):
    method_name = "base"
    score_name = "score"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__()
        self.in_dim = int(in_dim)
        self.config = config

    @abstractmethod
    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        raise NotImplementedError

    @abstractmethod
    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def summary(self) -> dict:
        return {
            "method": self.method_name,
            "score_name": self.score_name,
            "in_dim": self.in_dim,
            "config": self.config.to_dict(),
        }
