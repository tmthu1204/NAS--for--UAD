from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class UsadArchConfig:
    """
    Paper/repo-style USAD family config for raw SWaT source-only runs.

    This keeps the original family structure:
    - one MLP encoder
    - two MLP decoders
    - adversarial two-loss training
    - anomaly score = alpha * MSE(x, w1) + beta * MSE(x, w3)
    """

    window_length: int = 12
    downsample: int = 5
    latent_size: int = 1200
    hidden_scale: float = 1.0
    batch_size: int = 128
    max_epoch: int = 70
    valid_ratio: float = 0.2
    lr: float = 1e-3
    stride: int = 1
    score_alpha: float = 0.5
    score_beta: float = 0.5


def _paper_latent_size(window_length: int) -> int:
    # Match the upstream SWaT notebook: z_size = window_size * hidden_size, hidden_size=100.
    return max(1, int(window_length) * 100)


def get_fixed_paper_usad_arch(window_length: int = 12, downsample: int = 5) -> UsadArchConfig:
    return UsadArchConfig(
        window_length=window_length,
        downsample=downsample,
        latent_size=_paper_latent_size(window_length),
    )


def sample_usad_arch(window_length: int = 12, downsample: int = 5) -> UsadArchConfig:
    base_latent = _paper_latent_size(window_length)
    latent = random.choice(
        [
            max(1, int(round(base_latent * 0.75))),
            max(1, int(round(base_latent * 1.00))),
            max(1, int(round(base_latent * 1.25))),
        ]
    )
    hidden_scale = random.choice([0.75, 1.0, 1.25])
    return UsadArchConfig(
        window_length=window_length,
        downsample=downsample,
        latent_size=latent,
        hidden_scale=hidden_scale,
    )


def _hidden_dims(w_size: int, arch: UsadArchConfig) -> Tuple[int, int]:
    # Keep the original USAD macro shape fixed:
    # encoder widths follow the paper-style in_size -> in_size/2 -> in_size/4 -> latent,
    # while partial NAS only scales this hidden width envelope.
    h1 = max(1, int(round(w_size * 0.5 * arch.hidden_scale)))
    h2 = max(1, int(round(w_size * 0.25 * arch.hidden_scale)))
    if h2 > h1:
        h1, h2 = h2, h1
    return h1, h2


class Encoder(nn.Module):
    def __init__(self, in_size: int, latent_size: int, hidden1: int, hidden2: int):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, latent_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(w))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        return out


class Decoder(nn.Module):
    def __init__(self, latent_size: int, out_size: int, hidden1: int, hidden2: int):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, hidden2)
        self.linear2 = nn.Linear(hidden2, hidden1)
        self.linear3 = nn.Linear(hidden1, out_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        out = self.sigmoid(self.linear3(out))
        return out


class UsadModel(nn.Module):
    def __init__(self, w_size: int, arch: UsadArchConfig):
        super().__init__()
        self.arch = arch
        h1, h2 = _hidden_dims(w_size, arch)
        self.encoder = Encoder(w_size, arch.latent_size, h1, h2)
        self.decoder1 = Decoder(arch.latent_size, w_size, h1, h2)
        self.decoder2 = Decoder(arch.latent_size, w_size, h1, h2)

    def forward(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return {
            "z": z,
            "w1": w1,
            "w2": w2,
            "w3": w3,
        }

    def losses(self, batch: torch.Tensor, epoch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(batch)
        n = max(1, int(epoch_idx))
        mse1 = torch.mean((batch - out["w1"]) ** 2)
        mse2 = torch.mean((batch - out["w2"]) ** 2)
        mse3 = torch.mean((batch - out["w3"]) ** 2)
        loss1 = (1.0 / n) * mse1 + (1.0 - 1.0 / n) * mse3
        loss2 = (1.0 / n) * mse2 - (1.0 - 1.0 / n) * mse3
        return loss1, loss2

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.forward(batch)
        s1 = torch.mean((batch - out["w1"]) ** 2, dim=1)
        s2 = torch.mean((batch - out["w3"]) ** 2, dim=1)
        return self.arch.score_alpha * s1 + self.arch.score_beta * s2


def _window_loader(windows: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 2:
        raise ValueError(f"Expected flattened windows [N, W*C], got shape {windows.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(windows))
    ds = TensorDataset(tensor)
    return DataLoader(
        ds,
        batch_size=min(batch_size, max(1, len(ds))),
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def score_usad_windows(
    model: UsadModel,
    windows: np.ndarray,
    device: str,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    dl = _window_loader(windows, batch_size=batch_size, shuffle=False)
    scores = []
    for (xb,) in dl:
        xb = xb.to(device)
        scores.append(model.anomaly_score(xb).detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


@torch.no_grad()
def validate_usad_on_windows(
    model: UsadModel,
    windows: np.ndarray,
    device: str,
    *,
    batch_size: int,
    epoch_idx: int = 1,
) -> Dict[str, float]:
    model.eval()
    dl = _window_loader(windows, batch_size=batch_size, shuffle=False)
    loss1_vals: List[float] = []
    loss2_vals: List[float] = []
    score_vals: List[float] = []
    for (xb,) in dl:
        xb = xb.to(device)
        loss1, loss2 = model.losses(xb, epoch_idx=epoch_idx)
        loss1_vals.append(float(loss1.detach().cpu().item()))
        loss2_vals.append(float(loss2.detach().cpu().item()))
        score_vals.append(float(model.anomaly_score(xb).detach().mean().cpu().item()))
    return {
        "val_loss1": float(np.mean(loss1_vals)) if loss1_vals else float("inf"),
        "val_loss2": float(np.mean(loss2_vals)) if loss2_vals else float("inf"),
        "val_score": float(np.mean(score_vals)) if score_vals else float("inf"),
    }


def train_usad_source(
    model: UsadModel,
    train_windows: np.ndarray,
    val_windows: Optional[np.ndarray],
    device: str,
    *,
    arch: UsadArchConfig,
    epochs: Optional[int] = None,
    patience: int = 5,
    shuffle: bool = False,
    use_early_stopping: bool = False,
    restore_best_state: bool = False,
) -> Dict[str, List[float]]:
    model.to(device)
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=arch.lr,
    )
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=arch.lr,
    )

    train_dl = _window_loader(train_windows, batch_size=arch.batch_size, shuffle=shuffle)

    history = {
        "train_loss1": [],
        "train_loss2": [],
        "train_score": [],
        "val_loss1": [],
        "val_loss2": [],
        "val_score": [],
    }

    n_epochs = epochs if epochs is not None else arch.max_epoch
    best_obj = float("inf")
    best_state = None
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        batch_loss1: List[float] = []
        batch_loss2: List[float] = []
        batch_score: List[float] = []

        for (xb,) in train_dl:
            xb = xb.to(device)

            loss1, _ = model.losses(xb, epoch_idx=epoch + 1)
            opt1.zero_grad()
            loss1.backward()
            opt1.step()

            _, loss2 = model.losses(xb, epoch_idx=epoch + 1)
            opt2.zero_grad()
            loss2.backward()
            opt2.step()

            batch_loss1.append(float(loss1.detach().cpu().item()))
            batch_loss2.append(float(loss2.detach().cpu().item()))
            batch_score.append(float(model.anomaly_score(xb).detach().mean().cpu().item()))

        history["train_loss1"].append(float(np.mean(batch_loss1)) if batch_loss1 else float("inf"))
        history["train_loss2"].append(float(np.mean(batch_loss2)) if batch_loss2 else float("inf"))
        history["train_score"].append(float(np.mean(batch_score)) if batch_score else float("inf"))

        if val_windows is not None and len(val_windows) >= 1:
            val_stats = validate_usad_on_windows(
                model,
                val_windows,
                device=device,
                batch_size=arch.batch_size,
                epoch_idx=epoch + 1,
            )
            history["val_loss1"].append(val_stats["val_loss1"])
            history["val_loss2"].append(val_stats["val_loss2"])
            history["val_score"].append(val_stats["val_score"])
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
                if use_early_stopping and stale >= patience:
                    break

    if restore_best_state and best_state is not None:
        model.load_state_dict(best_state)

    return history


# Backward-compatible aliases used by earlier experiments.
score_usad_series = score_usad_windows
validate_usad_on_series = validate_usad_on_windows
