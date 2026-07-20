from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.drocc_head import DROCCHead

from .base import OneClassBackend, OneClassConfig


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


class DROCCHeadBackend(OneClassBackend):
    method_name = "drocc_head"
    score_name = "drocc_anomaly_logit"

    def __init__(self, in_dim: int, config: OneClassConfig):
        super().__init__(in_dim=in_dim, config=config)
        self.model = None

    def _build_model(self, device: str):
        self.model = DROCCHead(
            in_dim=self.in_dim,
            hidden_dim=max(1, int(self.config.drocc_hidden_dim)),
            rep_dim=max(2, int(self.config.drocc_rep_dim)),
        ).to(device)
        return self.model

    def _make_adversarial_latents(self, base_latents: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DROCCHeadBackend model must be built before generating adversarial latents.")

        centered = base_latents - self.model.train_center
        latent_scale = float(centered.norm(dim=1).mean().detach().clamp_min(1e-4).item())
        radius = float(max(self.config.drocc_radius, 1e-4) * latent_scale)
        gamma = float(max(self.config.drocc_gamma, 1.0))
        step_size = float(max(self.config.drocc_adv_step_size, 1e-4) * latent_scale)
        steps = max(1, int(self.config.drocc_adv_steps))

        adv = self.model.random_shell_latents(base_latents, radius=radius, gamma=gamma).detach()
        targets = torch.ones(base_latents.shape[0], dtype=base_latents.dtype, device=base_latents.device)

        for _ in range(steps):
            adv.requires_grad_(True)
            adv_logits = self.model.logits_from_latent(adv)
            adv_loss = F.binary_cross_entropy_with_logits(adv_logits, targets)
            grad = torch.autograd.grad(adv_loss, adv, only_inputs=True)[0]
            adv = adv.detach() + step_size * _normalize_rows(grad.detach())
            adv = self.model.project_to_shell(base_latents, adv, radius=radius, gamma=gamma).detach()
        return adv

    def fit(self, Zs_np: np.ndarray, device: str, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        Zs = torch.tensor(np.asarray(Zs_np), dtype=torch.float32, device=device)
        model = self._build_model(device)
        model.init_center(Zs)

        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        n = Zs.shape[0]
        warmup_epochs = max(1, min(int(self.config.drocc_warmup_epochs), int(self.config.epochs)))
        adv_weight = float(max(self.config.drocc_adv_weight, 0.0))
        compactness_weight = float(max(self.config.drocc_compactness_weight, 0.0))

        for ep in range(self.config.epochs):
            self.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.config.batch_size):
                zb = Zs[perm[i:i + self.config.batch_size]]
                compact_loss, _, latents = model.compactness_loss(zb)

                if ep < warmup_epochs:
                    loss = compact_loss
                else:
                    normal_logits = model.logits_from_latent(latents)
                    normal_targets = torch.zeros_like(normal_logits)
                    adv_latents = self._make_adversarial_latents(latents.detach())
                    adv_logits = model.logits_from_latent(adv_latents)
                    adv_targets = torch.ones_like(adv_logits)

                    normal_loss = F.binary_cross_entropy_with_logits(normal_logits, normal_targets)
                    adv_loss = F.binary_cross_entropy_with_logits(adv_logits, adv_targets)
                    loss = normal_loss + adv_weight * adv_loss + compactness_weight * compact_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        return self

    def score_tensor(self, Z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DROCCHeadBackend must be fit before scoring.")
        return self.model(Z)

    def summary(self) -> dict:
        out = super().summary()
        if self.model is None:
            out["rep_dim"] = None
        else:
            out["rep_dim"] = int(self.model.rep_dim)
        out["radius"] = float(self.config.drocc_radius)
        out["gamma"] = float(self.config.drocc_gamma)
        out["adv_steps"] = int(self.config.drocc_adv_steps)
        out["adv_step_size"] = float(self.config.drocc_adv_step_size)
        out["warmup_epochs"] = int(self.config.drocc_warmup_epochs)
        out["adv_weight"] = float(self.config.drocc_adv_weight)
        out["compactness_weight"] = float(self.config.drocc_compactness_weight)
        return out
