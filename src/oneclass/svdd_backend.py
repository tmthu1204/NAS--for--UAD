from __future__ import annotations

import numpy as np
import torch
from sklearn.svm import OneClassSVM

from .base import OneClassBackend, OneClassConfig


class SVDDBackend(OneClassBackend):
    method_name = "svdd"
    score_name = "dist2_minus_radius2"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None
        self.register_buffer("support_vectors", torch.empty(0, in_dim))
        self.register_buffer("dual_coef", torch.empty(0))
        self.radius2 = 0.0
        self.center_norm = 0.0
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
        raise ValueError(f"Unsupported SVDD kernel: {self.config.ocsvm_kernel}")

    def _kernel_numpy(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        kernel = str(self.config.ocsvm_kernel).strip().lower()
        if kernel == "linear":
            return X @ Y.T

        gamma = float(self.resolved_gamma)
        gram = X @ Y.T

        if kernel == "rbf":
            x_norm = np.sum(X * X, axis=1, keepdims=True)
            y_norm = np.sum(Y * Y, axis=1, keepdims=True).T
            dist2 = np.clip(x_norm + y_norm - 2.0 * gram, a_min=0.0, a_max=None)
            return np.exp(-gamma * dist2)
        if kernel == "poly":
            return (gamma * gram + float(self.config.ocsvm_coef0)) ** int(self.config.ocsvm_degree)
        if kernel == "sigmoid":
            return np.tanh(gamma * gram + float(self.config.ocsvm_coef0))
        raise ValueError(f"Unsupported SVDD kernel: {self.config.ocsvm_kernel}")

    def _self_kernel_tensor(self, X: torch.Tensor) -> torch.Tensor:
        kernel = str(self.config.ocsvm_kernel).strip().lower()
        if kernel == "rbf":
            return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)
        if kernel == "linear":
            return (X * X).sum(dim=1)
        gram_diag = (X * X).sum(dim=1)
        gamma = float(self.resolved_gamma)
        if kernel == "poly":
            return (gamma * gram_diag + float(self.config.ocsvm_coef0)) ** int(self.config.ocsvm_degree)
        if kernel == "sigmoid":
            return torch.tanh(gamma * gram_diag + float(self.config.ocsvm_coef0))
        raise ValueError(f"Unsupported SVDD kernel: {self.config.ocsvm_kernel}")

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
        self.resolved_gamma = float(model._gamma)

        support_vectors = np.asarray(model.support_vectors_, dtype=np.float32)
        dual_coef = np.asarray(model.dual_coef_.reshape(-1), dtype=np.float32)
        support_idx = np.asarray(model.support_, dtype=np.int64)

        self.support_vectors = torch.tensor(support_vectors, dtype=torch.float32, device=device)
        self.dual_coef = torch.tensor(dual_coef, dtype=torch.float32, device=device)

        K_sv = self._kernel_numpy(support_vectors, support_vectors)
        alpha = dual_coef
        self.center_norm = float(alpha @ (K_sv @ alpha))

        K_full_sv = self._kernel_numpy(Z_fit, support_vectors)
        diag_full = np.diag(self._kernel_numpy(Z_fit, Z_fit)).astype(np.float32, copy=False)
        dist2_train = diag_full - 2.0 * (K_full_sv @ alpha) + self.center_norm

        C = 1.0 / (float(self.config.ocsvm_nu) * float(Z_fit.shape[0]))
        support_alpha = np.zeros(Z_fit.shape[0], dtype=np.float32)
        support_alpha[support_idx] = alpha
        eps = 1e-6
        margin_mask = (support_alpha > eps) & (support_alpha < (C - eps))

        if np.any(margin_mask):
            radius_candidates = dist2_train[margin_mask]
            self.radius2 = float(np.mean(radius_candidates))
        elif np.any(support_alpha > eps):
            self.radius2 = float(np.max(dist2_train[support_alpha > eps]))
        else:
            self.radius2 = float(np.max(dist2_train))

        self.radius2 = max(0.0, self.radius2)
        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("SVDDBackend must be fit before scoring.")
        support = self.support_vectors.to(device=Z.device, dtype=Z.dtype)
        alpha = self.dual_coef.to(device=Z.device, dtype=Z.dtype)
        kz = self._kernel_tensor(Z, support)
        dist2 = self._self_kernel_tensor(Z) - 2.0 * (kz @ alpha) + float(self.center_norm)
        return dist2 - float(self.radius2)

    def summary(self) -> dict:
        out = super().summary()
        out["support_count"] = int(self.dual_coef.numel())
        out["kernel"] = str(self.config.ocsvm_kernel).strip().lower()
        out["resolved_gamma"] = self.resolved_gamma
        out["radius2"] = float(self.radius2)
        out["center_norm"] = float(self.center_norm)
        out["solver_backend"] = "libsvm_oneclass_equivalent"
        return out
