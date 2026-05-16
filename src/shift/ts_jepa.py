from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.data.datasets import ArrayDataset
from src.models.tscnn import EncoderCNN


@dataclass
class TSJepaConfig:
    in_channels: int
    d_model: int = 128
    enc_filters: Tuple[int, ...] = (64, 96, 128)
    enc_kernels: Tuple[int, ...] = (7, 5, 3)
    enc_strides: Tuple[int, ...] = (1, 1, 1)
    enc_dilations: Tuple[int, ...] = (1, 2, 4)
    activation: str = "relu"
    predictor_hidden: int = 128
    predictor_layers: int = 1
    predictor_dropout: float = 0.1
    mask_ratio: float = 0.40
    mask_span: int = 8
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    ema_momentum: float = 0.99
    valid_ratio: float = 0.10
    num_workers: int = 0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class TSJepaEncoder(nn.Module):
    def __init__(self, cfg: TSJepaConfig):
        super().__init__()
        self.backbone = EncoderCNN(
            cfg.in_channels,
            list(cfg.enc_filters),
            list(cfg.enc_kernels),
            list(cfg.enc_strides),
            pool=None,
            activation=cfg.activation,
            dilations=list(cfg.enc_dilations),
        )
        self.proj = nn.Sequential(
            nn.Linear(cfg.enc_filters[-1], cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(x)  # [B, T, C]
        return self.proj(tokens)

    def forward_vector(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x).mean(dim=1)


class TSJepaPredictor(nn.Module):
    def __init__(self, cfg: TSJepaConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.predictor_hidden,
            num_layers=cfg.predictor_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.predictor_dropout if cfg.predictor_layers > 1 else 0.0,
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.predictor_hidden * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        ctx, _ = self.gru(tokens)
        return self.out(ctx)


def build_time_mask(
    batch_size: int,
    length: int,
    *,
    mask_ratio: float,
    mask_span: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    num_to_mask = max(1, int(round(length * mask_ratio)))
    span = max(1, min(mask_span, length))

    for b in range(batch_size):
        remaining = num_to_mask
        attempts = 0
        while remaining > 0 and attempts < length * 4:
            start = int(torch.randint(0, max(1, length - span + 1), (1,), device=device).item())
            end = min(length, start + min(span, remaining))
            mask[b, start:end] = True
            remaining = num_to_mask - int(mask[b].sum().item())
            attempts += 1
        if not mask[b].any():
            mask[b, 0] = True
    return mask


class TSJepaModel(nn.Module):
    def __init__(self, cfg: TSJepaConfig):
        super().__init__()
        self.cfg = cfg
        self.online_encoder = TSJepaEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = TSJepaPredictor(cfg)
        self._freeze_target()

    def _freeze_target(self):
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)

    def compute_loss(
        self,
        x: torch.Tensor,
        *,
        mask_ratio: Optional[float] = None,
        mask_span: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict]:
        mask_ratio = self.cfg.mask_ratio if mask_ratio is None else mask_ratio
        mask_span = self.cfg.mask_span if mask_span is None else mask_span

        mask = build_time_mask(
            batch_size=x.size(0),
            length=x.size(1),
            mask_ratio=mask_ratio,
            mask_span=mask_span,
            device=x.device,
        )
        x_masked = x.clone()
        x_masked[mask] = 0.0

        online_tokens = self.online_encoder.forward_tokens(x_masked)
        pred_tokens = self.predictor(online_tokens)
        with torch.no_grad():
            target_tokens = self.target_encoder.forward_tokens(x)

        mask_f = mask.unsqueeze(-1).float()
        diff = (pred_tokens - target_tokens).pow(2) * mask_f
        denom = mask_f.sum().clamp_min(1.0) * pred_tokens.size(-1)
        loss = diff.sum() / denom

        stats = {
            "loss": float(loss.detach().cpu().item()),
            "mask_fraction": float(mask.float().mean().detach().cpu().item()),
        }
        return loss, stats


def _split_dataset(dataset: ArrayDataset, valid_ratio: float, seed: int):
    if valid_ratio <= 0.0 or len(dataset) < 4:
        return dataset, None
    n_valid = max(1, int(round(len(dataset) * valid_ratio)))
    n_valid = min(n_valid, len(dataset) - 1)
    n_train = len(dataset) - n_valid
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_valid], generator=generator)


def _loss_on_loader(
    model: TSJepaModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            loss, _ = model.compute_loss(xb)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_ts_jepa(
    model: TSJepaModel,
    x_train: np.ndarray,
    *,
    device: Union[str, torch.device],
) -> List[Dict[str, float]]:
    if len(x_train) == 0:
        raise ValueError("x_train is empty.")

    cfg = model.cfg
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(device)
    model.to(device)

    dataset = ArrayDataset(np.asarray(x_train, dtype=np.float32))
    train_ds, valid_ds = _split_dataset(dataset, cfg.valid_ratio, cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=min(cfg.batch_size, len(valid_ds)),
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
        )

    params = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    optim = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        batch_losses = []
        for xb in train_loader:
            xb = xb.to(device)
            loss, stats = model.compute_loss(xb)
            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()
            model.update_target_encoder(cfg.ema_momentum)
            batch_losses.append(stats["loss"])

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        valid_loss = _loss_on_loader(model, valid_loader, device) if valid_loader is not None else float("nan")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "num_train_batches": int(len(train_loader)),
            }
        )

    return history


@torch.no_grad()
def extract_jepa_features(
    model: TSJepaModel,
    x_data: np.ndarray,
    *,
    device: Union[str, torch.device],
    batch_size: int = 256,
    use_target_encoder: bool = True,
) -> np.ndarray:
    if len(x_data) == 0:
        return np.zeros((0, model.cfg.d_model), dtype=np.float32)

    device = torch.device(device)
    model.to(device)
    model.eval()
    encoder = model.target_encoder if use_target_encoder else model.online_encoder

    loader = DataLoader(
        ArrayDataset(np.asarray(x_data, dtype=np.float32)),
        batch_size=min(batch_size, len(x_data)),
        shuffle=False,
        drop_last=False,
        num_workers=model.cfg.num_workers,
    )

    feats = []
    for xb in loader:
        xb = xb.to(device)
        feats.append(encoder.forward_vector(xb).detach().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)
