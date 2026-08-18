from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.dagmm_head import DAGMMHead


@dataclass
class DagmmArchConfig:
    """
    Paper-family DAGMM config for source-only UAD on flattened time windows.

    The framework skeleton stays fixed:
    - autoencoder compression network
    - reconstruction cues appended to the latent code
    - estimation network for GMM responsibilities
    - sample energy as anomaly score

    Partial NAS only searches family-local capacity and mixture knobs.
    """

    window_length: int = 10
    hidden_dim: int = 128
    latent_dim: int = 16
    est_hidden_dim: int = 64
    num_components: int = 3
    batch_size: int = 64
    max_epoch: int = 10
    valid_ratio: float = 0.2
    lr: float = 1e-3
    lambda_energy: float = 0.1
    lambda_cov_diag: float = 5e-3
    warmup_epochs: int = 2
    gradient_clip_norm: float = 5.0


def get_fixed_paper_dagmm_arch(window_length: int = 10) -> DagmmArchConfig:
    return DagmmArchConfig(window_length=window_length)


def sample_dagmm_arch(window_length: int = 10) -> DagmmArchConfig:
    return DagmmArchConfig(
        window_length=window_length,
        hidden_dim=random.choice([64, 128, 256]),
        latent_dim=random.choice([8, 16, 32]),
        est_hidden_dim=random.choice([32, 64, 128]),
        num_components=random.choice([2, 3, 4, 5]),
    )


def _window_loader(windows: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 2:
        raise ValueError(f"Expected flattened windows [N, W*C], got shape {windows.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(windows))
    ds = TensorDataset(tensor)
    return DataLoader(
        ds,
        batch_size=min(int(batch_size), max(1, len(ds))),
        shuffle=shuffle,
        drop_last=False,
    )


class DagmmModel(DAGMMHead):
    def __init__(self, in_dim: int, arch: DagmmArchConfig):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=max(1, int(arch.hidden_dim)),
            latent_dim=max(1, int(arch.latent_dim)),
            est_hidden_dim=max(1, int(arch.est_hidden_dim)),
            num_components=max(1, int(arch.num_components)),
        )
        self.arch = arch


@torch.no_grad()
def refresh_dagmm_mixture_params(
    model: DagmmModel,
    windows: np.ndarray,
    device: str,
    *,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    augmented_chunks: List[torch.Tensor] = []
    gamma_chunks: List[torch.Tensor] = []
    for (xb,) in _window_loader(windows, batch_size=batch_size, shuffle=False):
        xb = xb.to(device)
        augmented, _, _, _ = model.build_augmented_latent(xb)
        gamma = model.estimate_gamma(augmented)
        augmented_chunks.append(augmented.detach())
        gamma_chunks.append(gamma.detach())

    if not augmented_chunks:
        raise ValueError("Cannot refresh DAGMM mixture params with no windows.")

    augmented_all = torch.cat(augmented_chunks, dim=0)
    gamma_all = torch.cat(gamma_chunks, dim=0)
    phi, mu, var = model.compute_gmm_params(augmented_all, gamma_all)
    model.set_mixture_params(phi, mu, var)
    return {
        "phi_min": float(phi.min().detach().cpu().item()),
        "phi_max": float(phi.max().detach().cpu().item()),
        "var_min": float(var.min().detach().cpu().item()),
        "var_max": float(var.max().detach().cpu().item()),
    }


@torch.no_grad()
def score_dagmm_windows(
    model: DagmmModel,
    windows: np.ndarray,
    device: str,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores = []
    for (xb,) in _window_loader(windows, batch_size=batch_size, shuffle=False):
        xb = xb.to(device)
        scores.append(model(xb).detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


@torch.no_grad()
def validate_dagmm_on_windows(
    model: DagmmModel,
    windows: np.ndarray,
    device: str,
    *,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    dl = _window_loader(windows, batch_size=batch_size, shuffle=False)
    loss_vals: List[float] = []
    recon_vals: List[float] = []
    energy_vals: List[float] = []
    cov_vals: List[float] = []

    for (xb,) in dl:
        xb = xb.to(device)
        augmented, _, _, recon_mse = model.build_augmented_latent(xb)
        energy = model.energy_from_params(augmented, model.phi, model.mu, model.var)
        cov_diag_penalty = model.covariance_regularizer(model.var)
        loss = (
            recon_mse.mean()
            + float(model.arch.lambda_energy) * energy.mean()
            + float(model.arch.lambda_cov_diag) * cov_diag_penalty
        )
        loss_vals.append(float(loss.detach().cpu().item()))
        recon_vals.append(float(recon_mse.mean().detach().cpu().item()))
        energy_vals.append(float(energy.mean().detach().cpu().item()))
        cov_vals.append(float(cov_diag_penalty.detach().cpu().item()))

    return {
        "val_loss": float(np.mean(loss_vals)) if loss_vals else float("inf"),
        "val_recon_mse": float(np.mean(recon_vals)) if recon_vals else float("inf"),
        "val_score": float(np.mean(energy_vals)) if energy_vals else float("inf"),
        "val_cov_diag_penalty": float(np.mean(cov_vals)) if cov_vals else float("inf"),
    }


def train_dagmm_source(
    model: DagmmModel,
    train_windows: np.ndarray,
    val_windows: Optional[np.ndarray],
    device: str,
    *,
    arch: DagmmArchConfig,
    epochs: Optional[int] = None,
    patience: int = 5,
    shuffle: bool = True,
    use_early_stopping: bool = False,
    restore_best_state: bool = False,
) -> Dict[str, List[float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=arch.lr)
    train_dl = _window_loader(train_windows, batch_size=arch.batch_size, shuffle=shuffle)

    history = {
        "train_loss": [],
        "train_recon_mse": [],
        "train_score": [],
        "train_cov_diag_penalty": [],
        "val_loss": [],
        "val_recon_mse": [],
        "val_score": [],
        "val_cov_diag_penalty": [],
    }

    n_epochs = epochs if epochs is not None else arch.max_epoch
    warmup_epochs = max(0, min(int(arch.warmup_epochs), int(n_epochs)))
    best_obj = float("inf")
    best_state = None
    stale = 0

    for epoch in range(int(n_epochs)):
        model.train()
        loss_vals: List[float] = []
        recon_vals: List[float] = []
        energy_vals: List[float] = []
        cov_vals: List[float] = []

        for (xb,) in train_dl:
            xb = xb.to(device)
            augmented, _, _, recon_mse = model.build_augmented_latent(xb)
            recon_loss = recon_mse.mean()

            if epoch < warmup_epochs:
                energy = torch.zeros((), dtype=xb.dtype, device=xb.device)
                cov_diag_penalty = torch.zeros((), dtype=xb.dtype, device=xb.device)
                loss = recon_loss
            else:
                gamma = model.estimate_gamma(augmented)
                phi, mu, var = model.compute_gmm_params(augmented, gamma)
                energy = model.energy_from_params(augmented, phi, mu, var).mean()
                cov_diag_penalty = model.covariance_regularizer(var)
                loss = (
                    recon_loss
                    + float(arch.lambda_energy) * energy
                    + float(arch.lambda_cov_diag) * cov_diag_penalty
                )

            opt.zero_grad()
            loss.backward()
            if arch.gradient_clip_norm and arch.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(arch.gradient_clip_norm))
            opt.step()

            loss_vals.append(float(loss.detach().cpu().item()))
            recon_vals.append(float(recon_loss.detach().cpu().item()))
            energy_vals.append(float(energy.detach().cpu().item()))
            cov_vals.append(float(cov_diag_penalty.detach().cpu().item()))

        history["train_loss"].append(float(np.mean(loss_vals)) if loss_vals else float("inf"))
        history["train_recon_mse"].append(float(np.mean(recon_vals)) if recon_vals else float("inf"))
        history["train_score"].append(float(np.mean(energy_vals)) if energy_vals else float("inf"))
        history["train_cov_diag_penalty"].append(float(np.mean(cov_vals)) if cov_vals else float("inf"))

        refresh_dagmm_mixture_params(
            model,
            train_windows,
            device,
            batch_size=arch.batch_size,
        )

        if val_windows is not None and len(val_windows) >= 1:
            val_stats = validate_dagmm_on_windows(
                model,
                val_windows,
                device=device,
                batch_size=arch.batch_size,
            )
            history["val_loss"].append(val_stats["val_loss"])
            history["val_recon_mse"].append(val_stats["val_recon_mse"])
            history["val_score"].append(val_stats["val_score"])
            history["val_cov_diag_penalty"].append(val_stats["val_cov_diag_penalty"])
            current_obj = val_stats["val_score"]
        else:
            current_obj = history["train_score"][-1]

        if restore_best_state or use_early_stopping:
            if current_obj < best_obj:
                best_obj = current_obj
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if use_early_stopping and stale >= int(patience):
                    break

    if restore_best_state and best_state is not None:
        model.load_state_dict(best_state)
        refresh_dagmm_mixture_params(
            model,
            train_windows,
            device,
            batch_size=arch.batch_size,
        )

    return history


# Backward-compatible aliases matching the other family modules.
score_dagmm_series = score_dagmm_windows
validate_dagmm_on_series = validate_dagmm_on_windows
