from __future__ import annotations

import numpy as np
import torch
from sklearn.svm import OneClassSVM

from .base import OneClassBackend, OneClassConfig


class OneClassSVMBackend(OneClassBackend):
    method_name = "oneclass_svm"
    score_name = "neg_decision_function"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None
        self.register_buffer("support_vectors", torch.empty(0, in_dim))
        self.register_buffer("dual_coef", torch.empty(0))
        self.intercept = 0.0
        self.resolved_gamma = None

    def _parse_gamma(self):
        gamma = self.config.ocsvm_gamma
        if isinstance(gamma, str):
            gamma = gamma.strip().lower()
            if gamma in {"scale", "auto"}:
                return gamma
            return float(gamma)
        return float(gamma)

    def _kernel_tensor(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        kernel = str(self.config.ocsvm_kernel).strip().lower()
        if kernel == "linear":
            return X @ Y.T

        gamma = float(self.resolved_gamma)
        gram = X @ Y.T

        if kernel == "rbf":
            x_norm = (X * X).sum(dim=1, keepdim=True)
            y_norm = (Y * Y).sum(dim=1).unsqueeze(0)
            dist2 = torch.clamp(x_norm + y_norm - 2.0 * gram, min=0.0)
            return torch.exp(-gamma * dist2)
        if kernel == "poly":
            return (gamma * gram + float(self.config.ocsvm_coef0)) ** int(self.config.ocsvm_degree)
        if kernel == "sigmoid":
            return torch.tanh(gamma * gram + float(self.config.ocsvm_coef0))
        raise ValueError(f"Unsupported One-Class SVM kernel: {self.config.ocsvm_kernel}")

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        Z_fit = np.asarray(Zs_np, dtype=np.float32)
        model = OneClassSVM(
            kernel=str(self.config.ocsvm_kernel).strip().lower(),
            nu=float(self.config.ocsvm_nu),
            gamma=self._parse_gamma(),
            degree=int(self.config.ocsvm_degree),
            coef0=float(self.config.ocsvm_coef0),
        )
        model.fit(Z_fit)

        self.model = model
        self.intercept = float(model.intercept_[0])
        self.resolved_gamma = float(model._gamma)
        self.support_vectors = torch.tensor(
            np.asarray(model.support_vectors_, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self.dual_coef = torch.tensor(
            np.asarray(model.dual_coef_.reshape(-1), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("OneClassSVMBackend must be fit before scoring.")
        support = self.support_vectors.to(device=Z.device, dtype=Z.dtype)
        dual = self.dual_coef.to(device=Z.device, dtype=Z.dtype)
        kernel_vals = self._kernel_tensor(Z, support)
        decision = kernel_vals @ dual + self.intercept
        return -decision

    def summary(self) -> dict:
        out = super().summary()
        out["support_count"] = int(self.dual_coef.numel())
        out["kernel"] = str(self.config.ocsvm_kernel).strip().lower()
        out["resolved_gamma"] = self.resolved_gamma
        out["intercept"] = float(self.intercept)
        return out
