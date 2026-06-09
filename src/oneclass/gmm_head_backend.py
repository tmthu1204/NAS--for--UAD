from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from src.models.gmm_head import GMMHead

from .base import OneClassBackend, OneClassConfig


class GMMHeadBackend(OneClassBackend):
    method_name = "gmm_head"
    score_name = "gmm_neg_log_likelihood"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None
        self.final_gmm = None

    def _build_model(self, device: str, num_components: int):
        self.model = GMMHead(
            in_dim=self.in_dim,
            hidden_dim=max(1, int(self.config.gmm_hidden_dim)),
            rep_dim=max(1, int(self.config.gmm_rep_dim)),
            num_components=max(1, int(num_components)),
            covariance_type=str(self.config.gmm_covariance_type).strip().lower(),
        ).to(device)
        return self.model

    def _fit_gmm(self, H_np: np.ndarray, seed: int):
        n_samples = int(H_np.shape[0])
        n_components = max(1, min(int(self.config.gmm_components), n_samples))
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=str(self.config.gmm_covariance_type).strip().lower(),
            reg_covar=float(self.config.gmm_reg_covar),
            random_state=seed,
            n_init=3,
            init_params="kmeans",
        )
        gmm.fit(H_np)
        return gmm

    @torch.no_grad()
    def _refresh_gmm_params(self, Zs: torch.Tensor, seed: int):
        H_np = self.model.encode(Zs).detach().cpu().numpy().astype(np.float32, copy=False)
        self.final_gmm = self._fit_gmm(H_np, seed=seed)
        self.model.set_gmm_params(
            weights=self.final_gmm.weights_,
            means=self.final_gmm.means_,
            precisions_cholesky=self.final_gmm.precisions_cholesky_,
        )

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        Zs = torch.tensor(np.asarray(Zs_np), dtype=torch.float32, device=device)
        effective_components = max(1, min(int(self.config.gmm_components), int(Zs.shape[0])))
        model = self._build_model(device, num_components=effective_components)
        model.init_center(Zs)

        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        n = Zs.shape[0]
        warmup_epochs = max(1, min(int(self.config.gmm_warmup_epochs), int(self.config.epochs)))

        for ep in range(self.config.epochs):
            self.train()
            if ep >= warmup_epochs:
                self.eval()
                self._refresh_gmm_params(Zs, seed=seed)
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
        self._refresh_gmm_params(Zs, seed=seed)
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("GMMHeadBackend must be fit before scoring.")
        return self.model(Z)

    def summary(self) -> dict:
        out = super().summary()
        if self.model is None or self.final_gmm is None:
            out["rep_dim"] = None
            out["num_components"] = None
            out["covariance_type"] = None
            out["lower_bound"] = None
        else:
            out["rep_dim"] = int(self.model.rep_dim)
            out["num_components"] = int(self.final_gmm.n_components)
            out["covariance_type"] = str(self.final_gmm.covariance_type)
            out["lower_bound"] = float(self.final_gmm.lower_bound_)
        return out
