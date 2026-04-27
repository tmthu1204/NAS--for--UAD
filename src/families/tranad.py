from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TranADArchConfig:
    """
    Upstream-faithful TranAD family config for raw SMD source-only runs.

    The frozen core follows the original repo/paper:
    - one positional encoder over concatenated source + conditioning
    - one transformer encoder
    - two transformer decoders for the two self-conditioning phases
    - phase-weighted reconstruction loss across epochs
    - last-step forecasting MSE as anomaly score
    """

    window_length: int = 10
    ff_dim: int = 16
    dropout: float = 0.1
    encoder_layers: int = 1
    decoder_layers: int = 1
    batch_size: int = 128
    max_epoch: int = 5
    valid_ratio: float = 0.2
    lr: float = 1e-4


def get_fixed_paper_tranad_arch(window_length: int = 10) -> TranADArchConfig:
    return TranADArchConfig(window_length=window_length)


def sample_tranad_arch(window_length: int = 10) -> TranADArchConfig:
    return TranADArchConfig(
        window_length=window_length,
        ff_dim=random.choice([16, 32, 64]),
        dropout=random.choice([0.05, 0.10, 0.20]),
        encoder_layers=random.choice([1, 2]),
        decoder_layers=random.choice([1, 2]),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.double)
        position = torch.arange(0, max_len, dtype=torch.double).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, dtype=torch.double) * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TranADTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TranADTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranADModel(nn.Module):
    def __init__(self, feats: int, arch: TranADArchConfig):
        super().__init__()
        self.name = "TranAD"
        self.arch = arch
        self.lr = arch.lr
        self.batch = arch.batch_size
        self.n_feats = feats
        self.n_window = arch.window_length
        d_model = 2 * feats

        self.pos_encoder = PositionalEncoding(d_model, arch.dropout, self.n_window)
        encoder_layer = TranADTransformerEncoderLayer(
            d_model=d_model,
            nhead=feats,
            dim_feedforward=arch.ff_dim,
            dropout=arch.dropout,
        )
        decoder_layer1 = TranADTransformerDecoderLayer(
            d_model=d_model,
            nhead=feats,
            dim_feedforward=arch.ff_dim,
            dropout=arch.dropout,
        )
        decoder_layer2 = TranADTransformerDecoderLayer(
            d_model=d_model,
            nhead=feats,
            dim_feedforward=arch.ff_dim,
            dropout=arch.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, arch.encoder_layers)
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1, arch.decoder_layers)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, arch.decoder_layers)
        self.fcn = nn.Sequential(nn.Linear(d_model, feats), nn.Sigmoid())

    def encode(self, src: torch.Tensor, conditioning: torch.Tensor, tgt: torch.Tensor):
        src = torch.cat((src, conditioning), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        conditioning = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, conditioning, tgt)))
        conditioning = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, conditioning, tgt)))
        return x1, x2


def _window_loader(windows: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.as_tensor(np.ascontiguousarray(windows), dtype=torch.double)
    ds = TensorDataset(tensor)
    return DataLoader(
        ds,
        batch_size=min(batch_size, max(1, len(ds))),
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def score_tranad_windows(
    model: TranADModel,
    windows: np.ndarray,
    device: str,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    dl = _window_loader(windows, batch_size=batch_size, shuffle=False)
    scores: List[np.ndarray] = []
    for (xb,) in dl:
        xb = xb.to(device)
        local_bs = xb.shape[0]
        window = xb.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, model.n_feats)
        _, x2 = model(window, elem)
        mse = torch.mean((x2 - elem) ** 2, dim=2).squeeze(0)
        scores.append(mse.detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float64)


@torch.no_grad()
def validate_tranad_on_windows(
    model: TranADModel,
    windows: np.ndarray,
    device: str,
    *,
    batch_size: int,
    epoch_idx: int = 1,
) -> Dict[str, float]:
    model.eval()
    dl = _window_loader(windows, batch_size=batch_size, shuffle=False)
    phase1_vals: List[float] = []
    phase2_vals: List[float] = []
    loss_vals: List[float] = []
    score_vals: List[float] = []

    n = max(1, int(epoch_idx))
    for (xb,) in dl:
        xb = xb.to(device)
        local_bs = xb.shape[0]
        window = xb.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, model.n_feats)
        x1, x2 = model(window, elem)
        mse1 = torch.mean((x1 - elem) ** 2)
        mse2 = torch.mean((x2 - elem) ** 2)
        loss = (1.0 / n) * mse1 + (1.0 - 1.0 / n) * mse2

        phase1_vals.append(float(mse1.detach().cpu().item()))
        phase2_vals.append(float(mse2.detach().cpu().item()))
        loss_vals.append(float(loss.detach().cpu().item()))
        score_vals.append(float(torch.mean((x2 - elem) ** 2, dim=2).mean().detach().cpu().item()))

    return {
        "val_phase1_mse": float(np.mean(phase1_vals)) if phase1_vals else float("inf"),
        "val_phase2_mse": float(np.mean(phase2_vals)) if phase2_vals else float("inf"),
        "val_loss": float(np.mean(loss_vals)) if loss_vals else float("inf"),
        "val_score": float(np.mean(score_vals)) if score_vals else float("inf"),
    }


def train_tranad_source(
    model: TranADModel,
    train_windows: np.ndarray,
    val_windows: Optional[np.ndarray],
    device: str,
    *,
    arch: TranADArchConfig,
    epochs: Optional[int] = None,
    patience: int = 5,
    shuffle: bool = False,
    use_early_stopping: bool = False,
    restore_best_state: bool = False,
) -> Dict[str, List[float]]:
    model.to(device)
    model.double()

    optimizer = torch.optim.AdamW(model.parameters(), lr=arch.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    train_dl = _window_loader(train_windows, batch_size=arch.batch_size, shuffle=shuffle)

    history = {
        "train_phase1_mse": [],
        "train_phase2_mse": [],
        "train_loss": [],
        "train_score": [],
        "val_phase1_mse": [],
        "val_phase2_mse": [],
        "val_loss": [],
        "val_score": [],
    }

    n_epochs = epochs if epochs is not None else arch.max_epoch
    best_obj = float("inf")
    best_state = None
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        phase1_vals: List[float] = []
        phase2_vals: List[float] = []
        loss_vals: List[float] = []
        score_vals: List[float] = []

        n = epoch + 1
        for (xb,) in train_dl:
            xb = xb.to(device)
            local_bs = xb.shape[0]
            window = xb.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, model.n_feats)
            x1, x2 = model(window, elem)

            mse1 = torch.mean((x1 - elem) ** 2)
            mse2 = torch.mean((x2 - elem) ** 2)
            loss = (1.0 / n) * mse1 + (1.0 - 1.0 / n) * mse2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            phase1_vals.append(float(mse1.detach().cpu().item()))
            phase2_vals.append(float(mse2.detach().cpu().item()))
            loss_vals.append(float(loss.detach().cpu().item()))
            score_vals.append(float(torch.mean((x2 - elem) ** 2, dim=2).mean().detach().cpu().item()))

        scheduler.step()

        history["train_phase1_mse"].append(float(np.mean(phase1_vals)) if phase1_vals else float("inf"))
        history["train_phase2_mse"].append(float(np.mean(phase2_vals)) if phase2_vals else float("inf"))
        history["train_loss"].append(float(np.mean(loss_vals)) if loss_vals else float("inf"))
        history["train_score"].append(float(np.mean(score_vals)) if score_vals else float("inf"))

        if val_windows is not None and len(val_windows) >= 1:
            val_stats = validate_tranad_on_windows(
                model,
                val_windows,
                device=device,
                batch_size=arch.batch_size,
                epoch_idx=epoch + 1,
            )
            history["val_phase1_mse"].append(val_stats["val_phase1_mse"])
            history["val_phase2_mse"].append(val_stats["val_phase2_mse"])
            history["val_loss"].append(val_stats["val_loss"])
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


score_tranad_series = score_tranad_windows
validate_tranad_on_series = validate_tranad_on_windows
