from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader

from src.data.omni_smd import SlidingWindowDataset


@dataclass
class OmniArchConfig:
    """
    Paper-faithful OmniAnomaly family config for source-only UAD on raw SMD.

    Defaults follow the published model family as closely as possible in this
    PyTorch refactor:
    - GRU recurrent cell
    - connected q(z_t | h_t, z_{t-1})
    - connected p(z_t | z_{t-1}) with identity-style transition
    - planar normalizing flow on the posterior
    - last-point reconstruction-probability anomaly scoring
    """

    window_length: int = 100
    z_dim: int = 3
    rnn_hidden: int = 500
    dense_dim: int = 500
    use_connected_z_q: bool = True
    use_connected_z_p: bool = True
    posterior_flow: str = "nf"
    nf_layers: int = 20
    batch_size: int = 50
    max_epoch: int = 20
    valid_ratio: float = 0.3
    lr: float = 1e-3
    l2_reg: float = 1e-4
    gradient_clip_norm: float = 10.0
    test_n_z: int = 1
    stride: int = 1


def get_fixed_paper_omni_arch(window_length: int = 100) -> OmniArchConfig:
    return OmniArchConfig(window_length=window_length)


def sample_omni_arch(window_length: int = 100) -> OmniArchConfig:
    return OmniArchConfig(
        window_length=window_length,
        z_dim=random.choice([3, 5, 8]),
        rnn_hidden=random.choice([300, 500, 700]),
        dense_dim=random.choice([300, 500, 700]),
        use_connected_z_q=True,
        use_connected_z_p=True,
        posterior_flow="nf",
        nf_layers=random.choice([10, 20, 30]),
    )


class PlanarFlow(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def _u_hat(self):
        wu = torch.dot(self.w, self.u)
        m = -1.0 + F.softplus(wu)
        w_norm_sq = torch.sum(self.w ** 2) + 1e-8
        return self.u + ((m - wu) * self.w / w_norm_sq)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [..., D]
        Returns:
            z_new: [..., D]
            log_abs_det: [...]
        """
        u_hat = self._u_hat()
        linear = torch.matmul(z, self.w) + self.b  # [...]
        h = torch.tanh(linear)
        z_new = z + h.unsqueeze(-1) * u_hat

        psi = (1.0 - torch.tanh(linear) ** 2).unsqueeze(-1) * self.w
        det_term = 1.0 + torch.sum(psi * u_hat, dim=-1)
        log_abs_det = torch.log(torch.abs(det_term) + 1e-8)
        return z_new, log_abs_det


class OmniAnomalyModel(nn.Module):
    def __init__(self, x_dim: int, arch: OmniArchConfig, eps: float = 1e-4):
        super().__init__()
        self.x_dim = x_dim
        self.arch = arch
        self.eps = eps

        self.q_rnn = nn.GRU(
            input_size=x_dim,
            hidden_size=arch.rnn_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.p_rnn = nn.GRU(
            input_size=arch.z_dim,
            hidden_size=arch.rnn_hidden,
            num_layers=1,
            batch_first=True,
        )

        q_in = arch.rnn_hidden + (arch.z_dim if arch.use_connected_z_q else 0)
        self.q_hidden = nn.Sequential(
            nn.Linear(q_in, arch.dense_dim),
            nn.ReLU(),
        )
        self.q_mu = nn.Linear(arch.dense_dim, arch.z_dim)
        self.q_logvar = nn.Linear(arch.dense_dim, arch.z_dim)

        if arch.use_connected_z_p:
            self.p_hidden = nn.Sequential(
                nn.Linear(arch.z_dim, arch.dense_dim),
                nn.ReLU(),
            )
            self.p_mu = nn.Linear(arch.dense_dim, arch.z_dim)
            self.p_logvar = nn.Linear(arch.dense_dim, arch.z_dim)
        else:
            self.p_hidden = None
            self.p_mu = None
            self.p_logvar = None

        self.dec_hidden = nn.Sequential(
            nn.Linear(arch.rnn_hidden, arch.dense_dim),
            nn.ReLU(),
        )
        self.x_mu = nn.Linear(arch.dense_dim, x_dim)
        self.x_logstd = nn.Linear(arch.dense_dim, x_dim)

        if arch.posterior_flow == "nf" and arch.nf_layers > 0:
            self.posterior_flows = nn.ModuleList([PlanarFlow(arch.z_dim) for _ in range(arch.nf_layers)])
        else:
            self.posterior_flows = nn.ModuleList()

    def _posterior_base_step(self, h_t: torch.Tensor, z_prev_q: torch.Tensor):
        if self.arch.use_connected_z_q:
            q_in = torch.cat([h_t, z_prev_q], dim=-1)
        else:
            q_in = h_t
        q_h = self.q_hidden(q_in)
        mu_q = self.q_mu(q_h)
        logvar_q = self.q_logvar(q_h).clamp(min=-8.0, max=8.0)
        std_q = torch.exp(0.5 * logvar_q).clamp_min(self.eps)
        eps = torch.randn_like(std_q)
        z0_t = mu_q + eps * std_q
        return z0_t, mu_q, logvar_q

    def _apply_posterior_flow(self, z0: torch.Tensor):
        z = z0
        sum_log_det = torch.zeros(z0.shape[:-1], dtype=z0.dtype, device=z0.device)
        for flow in self.posterior_flows:
            z, log_det = flow(z)
            sum_log_det = sum_log_det + log_det
        return z, sum_log_det

    def _prior_params(self, z: torch.Tensor):
        if not self.arch.use_connected_z_p or self.p_hidden is None:
            mu_p = torch.zeros_like(z)
            logvar_p = torch.zeros_like(z)
            return mu_p, logvar_p

        z_prev = torch.zeros_like(z)
        if z.size(1) > 1:
            z_prev[:, 1:, :] = z[:, :-1, :]

        p_h = self.p_hidden(z_prev)
        mu_p = self.p_mu(p_h)
        logvar_p = self.p_logvar(p_h).clamp(min=-8.0, max=8.0)
        return mu_p, logvar_p

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, C], got {tuple(x.shape)}")

        B, T, C = x.shape
        if C != self.x_dim:
            raise ValueError(f"Expected x_dim={self.x_dim}, got C={C}")

        h_seq, _ = self.q_rnn(x)

        z_prev_q = torch.zeros(B, self.arch.z_dim, device=x.device, dtype=x.dtype)
        z0_list: List[torch.Tensor] = []
        mu_q_list: List[torch.Tensor] = []
        logvar_q_list: List[torch.Tensor] = []

        for t in range(T):
            z0_t, mu_q_t, logvar_q_t = self._posterior_base_step(h_seq[:, t, :], z_prev_q)
            z0_list.append(z0_t)
            mu_q_list.append(mu_q_t)
            logvar_q_list.append(logvar_q_t)
            z_prev_q = z0_t

        z0 = torch.stack(z0_list, dim=1)
        mu_q = torch.stack(mu_q_list, dim=1)
        logvar_q = torch.stack(logvar_q_list, dim=1)
        z, log_det_sum = self._apply_posterior_flow(z0)
        mu_p, logvar_p = self._prior_params(z)

        dec_h, _ = self.p_rnn(z)
        dec_h = self.dec_hidden(dec_h)
        x_mu = self.x_mu(dec_h)
        x_std = F.softplus(self.x_logstd(dec_h)) + self.eps

        recon_dist = Normal(loc=x_mu, scale=x_std)
        recon_log_prob = recon_dist.log_prob(x).sum(dim=-1)  # [B, T]
        score_last = -recon_log_prob[:, -1]

        return {
            "z0": z0,
            "z": z,
            "log_det_sum": log_det_sum,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "x_mu": x_mu,
            "x_std": x_std,
            "recon_log_prob": recon_log_prob,
            "score_last": score_last,
        }

    def loss(self, x: torch.Tensor):
        out = self.forward(x)

        z0 = out["z0"]
        z = out["z"]
        mu_q = out["mu_q"]
        logvar_q = out["logvar_q"]
        mu_p = out["mu_p"]
        logvar_p = out["logvar_p"]
        recon_log_prob = out["recon_log_prob"]
        log_det_sum = out["log_det_sum"]

        std_q = torch.exp(0.5 * logvar_q).clamp_min(self.eps)
        std_p = torch.exp(0.5 * logvar_p).clamp_min(self.eps)

        q0_dist = Normal(mu_q, std_q)
        p_dist = Normal(mu_p, std_p)

        log_q0 = q0_dist.log_prob(z0).sum(dim=-1)  # [B, T]
        log_q = log_q0 - log_det_sum
        log_p = p_dist.log_prob(z).sum(dim=-1)  # [B, T]

        elbo_per_window = recon_log_prob.sum(dim=-1) + log_p.sum(dim=-1) - log_q.sum(dim=-1)
        loss = -elbo_per_window.mean()

        stats = {
            "loss": float(loss.detach().cpu().item()),
            "mean_elbo": float(elbo_per_window.detach().mean().cpu().item()),
            "mean_score_last": float(out["score_last"].detach().mean().cpu().item()),
        }
        return loss, stats


@torch.no_grad()
def score_omni_series(
    model: OmniAnomalyModel,
    series: np.ndarray,
    device: str,
    *,
    batch_size: int,
    window_length: int,
    stride: int = 1,
    n_z: int = 1,
) -> np.ndarray:
    model.eval()
    ds = SlidingWindowDataset(series, window_length=window_length, stride=stride)
    dl = DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=False, drop_last=False)
    scores = []
    for xb in dl:
        xb = xb.to(device)
        sample_scores = []
        for _ in range(max(1, int(n_z))):
            out = model(xb)
            sample_scores.append(out["score_last"].detach().cpu())
        mean_score = torch.stack(sample_scores, dim=0).mean(dim=0)
        scores.append(mean_score.numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


@torch.no_grad()
def validate_omni_on_series(
    model: OmniAnomalyModel,
    series: np.ndarray,
    device: str,
    *,
    batch_size: int,
    window_length: int,
    stride: int = 1,
    n_z: int = 1,
) -> Dict[str, float]:
    model.eval()
    ds = SlidingWindowDataset(series, window_length=window_length, stride=stride)
    dl = DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=False, drop_last=False)
    losses = []
    mean_scores = []
    for xb in dl:
        xb = xb.to(device)
        loss, stats = model.loss(xb)
        losses.append(float(loss.detach().cpu().item()))
        sample_scores = []
        for _ in range(max(1, int(n_z))):
            out = model(xb)
            sample_scores.append(float(out["score_last"].detach().mean().cpu().item()))
        mean_scores.append(float(np.mean(sample_scores)))
    return {
        "val_loss": float(np.mean(losses)) if losses else float("inf"),
        "val_score_last": float(np.mean(mean_scores)) if mean_scores else float("inf"),
    }


def train_omni_source(
    model: OmniAnomalyModel,
    train_series: np.ndarray,
    val_series: Optional[np.ndarray],
    device: str,
    *,
    arch: OmniArchConfig,
    epochs: Optional[int] = None,
    patience: int = 5,
) -> Dict[str, List[float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=arch.lr, weight_decay=arch.l2_reg)

    ds_train = SlidingWindowDataset(train_series, window_length=arch.window_length, stride=arch.stride)
    train_dl = DataLoader(
        ds_train,
        batch_size=min(arch.batch_size, max(1, len(ds_train))),
        shuffle=True,
        drop_last=False,
    )

    history = {
        "train_loss": [],
        "train_score_last": [],
        "val_loss": [],
        "val_score_last": [],
    }

    n_epochs = epochs if epochs is not None else arch.max_epoch
    best_obj = float("inf")
    best_state = None
    stale = 0

    for _ in range(n_epochs):
        model.train()
        batch_losses = []
        batch_scores = []
        for xb in train_dl:
            xb = xb.to(device)
            loss, stats = model.loss(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=arch.gradient_clip_norm)
            opt.step()

            batch_losses.append(stats["loss"])
            batch_scores.append(stats["mean_score_last"])

        history["train_loss"].append(float(np.mean(batch_losses)) if batch_losses else float("inf"))
        history["train_score_last"].append(float(np.mean(batch_scores)) if batch_scores else float("inf"))

        if val_series is not None and len(val_series) >= arch.window_length:
            val_stats = validate_omni_on_series(
                model,
                val_series,
                device=device,
                batch_size=arch.batch_size,
                window_length=arch.window_length,
                stride=arch.stride,
            )
            history["val_loss"].append(val_stats["val_loss"])
            history["val_score_last"].append(val_stats["val_score_last"])
            current_obj = val_stats["val_score_last"]
        else:
            current_obj = history["train_score_last"][-1]

        if current_obj < best_obj:
            best_obj = current_obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
