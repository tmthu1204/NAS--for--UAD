"""
Top-level orchestrator: preprocessing, TS-TCC pretrain, (optional) pseudo-labeling,
AdaptNAS search, final training.
Current default (non --uad): TS-TCC features + DeepSVDD anomaly score -> unsupervised weighting
-> AdaptNAS bilevel search -> final training.
"""
import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import ArrayDataset
from src.ts_tcc.trainer.trainer import TSTrainer
from src.adaptnas.search_space import sample_arch
from src.adaptnas.trainer import train_bilevel, quick_validate

from src.models.tscnn import EncoderCNN
from src.models.transformer import ARTransformer
from src.models.classifier import MLP
from src.models.discriminator import DomainDiscriminator

from src.models.deepsvdd import DeepSVDD

from src.utils.metrics import (
    compute_ap_auroc, pot_threshold, f1_at_threshold, best_f1, event_f1_and_delay
)

from sklearn.metrics import roc_auc_score


# ---------------- Seed / Determinism ----------------
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Console encoding ----------------
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------- Utils ----------------
def load_npz_if_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    if "X" not in data:
        raise ValueError(f"{path} missing key 'X'")
    X = data["X"]
    y = data["y"] if "y" in data else None
    return X, y


def fix_length(X, window=128):
    X_fixed = []
    for x in X:
        if x.ndim != 2:
            x = np.array(x).reshape(-1, x.shape[-1])
        T, C = x.shape
        if T < window:
            pad = np.zeros((window - T, C))
            x = np.concatenate([x, pad], axis=0)
        elif T > window:
            x = x[:window]
        X_fixed.append(x)
    return np.stack(X_fixed)


def load_all_smd_for_pretrain(root_dir="data/smd", window=128):
    import glob
    all_X = []
    n_machines = 0

    for mdir in sorted(glob.glob(os.path.join(root_dir, "machine-*"))):
        used_any = False
        for fname in ["source.npz", "target.npz"]:
            npz_path = os.path.join(mdir, fname)
            if not os.path.exists(npz_path):
                continue
            Xm, _ = load_npz_if_exists(npz_path)
            Xm = fix_length(Xm, window=window)
            all_X.append(Xm)
            used_any = True
        if used_any:
            n_machines += 1

    if not all_X:
        raise RuntimeError(f"No SMD source/target npz found under {root_dir}")
    X_all = np.concatenate(all_X, axis=0)
    print(f"[INFO] Multi-machine TS-TCC pretrain: {X_all.shape[0]} windows from {n_machines} machines.")
    return X_all


def build_validation(ds_source, ds_target, beta=0.5, m=200, bs=64, seed=42, fixed_s_idx=None):
    """
    Build:
      - val_src: source only
      - val_tgt: target only
      - val_hybrid: concat source+target (compat for step_upper)
    return: (val_hybrid, s_idx, val_src, val_tgt, beta_eff)
    """
    rng = np.random.RandomState(seed)
    ns, nt = len(ds_source), len(ds_target)
    ms = max(1, int(beta * m))
    mt = max(1, m - ms)

    if fixed_s_idx is None:
        s_idx = rng.choice(ns, size=min(ms, ns), replace=False)
    else:
        s_idx = fixed_s_idx[:min(ms, len(fixed_s_idx))]

    t_all = np.arange(nt)
    rng.shuffle(t_all)
    t_idx = t_all[:min(mt, nt)]

    Xs_val = np.stack([ds_source[i][0].numpy() for i in s_idx])
    ys_val = np.array([ds_source[i][1].item() for i in s_idx])

    Xt_val = np.stack([ds_target[i][0].numpy() for i in t_idx])
    yt_val = np.array([ds_target[i][1].item() for i in t_idx])

    val_src = DataLoader(ArrayDataset(Xs_val, ys_val), batch_size=bs, shuffle=False)
    val_tgt = DataLoader(ArrayDataset(Xt_val, yt_val), batch_size=bs, shuffle=False)

    Xhyb = np.concatenate([Xs_val, Xt_val], axis=0)
    yhyb = np.concatenate([ys_val, yt_val], axis=0)
    val_hybrid = DataLoader(ArrayDataset(Xhyb, yhyb), batch_size=bs, shuffle=False)

    beta_eff = len(ys_val) / max(1, len(yhyb))
    return val_hybrid, s_idx, val_src, val_tgt, beta_eff


# ---------------- Candidate model ----------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class CandidateModel(torch.nn.Module):
    def __init__(self, in_ch, arch, num_classes=2):
        super().__init__()
        self.seq_type = arch.seq_type

        self.encoder = EncoderCNN(
            in_ch,
            arch.enc_filters,
            arch.enc_kernels,
            arch.enc_strides,
            pool=arch.enc_pool,
            activation=arch.enc_activation,
            dilations=arch.enc_dilations,
        )

        last = arch.enc_filters[-1]
        self.to_d = nn.Conv1d(last, arch.d_model, kernel_size=1)

        if self.seq_type == "transformer":
            self.sequence = ARTransformer(
                d_model=arch.d_model,
                nhead=arch.seq_heads,
                num_layers=arch.seq_layers,
                dim_feedforward=arch.seq_hidden,
            )
        elif self.seq_type == "gru":
            self.sequence = nn.GRU(
                input_size=arch.d_model,
                hidden_size=arch.d_model,
                num_layers=arch.seq_layers,
                batch_first=True,
                bidirectional=False,
            )
        elif self.seq_type == "tcn":
            blocks = []
            in_c = arch.d_model
            for l in range(arch.seq_layers):
                dilation = arch.seq_dilation ** l
                blocks.append(
                    TCNBlock(in_c, arch.d_model, kernel_size=arch.seq_kernel, dilation=dilation)
                )
                in_c = arch.d_model
            self.sequence = nn.Sequential(*blocks)
        else:
            raise ValueError(f"Unknown seq_type: {self.seq_type}")

        self.classifier = MLP(arch.d_model, arch.clf_layers, arch.clf_units, out=num_classes)
        self.discriminator = DomainDiscriminator(arch.d_model)

        num_ops = len(arch.__dict__)
        self.arch_params = nn.Parameter(torch.randn(num_ops))

    def forward_features(self, x):
        z = self.encoder(x)      # (B, T, C_enc)
        z = z.transpose(1, 2)    # (B, C_enc, T)
        z = self.to_d(z)         # (B, d_model, T)

        if self.seq_type == "tcn":
            h = self.sequence(z)         # (B, d_model, T)
            f = h.mean(dim=2)            # (B, d_model)
        else:
            z_seq = z.transpose(1, 2)    # (B, T, d_model)
            if self.seq_type == "transformer":
                out = self.sequence(z_seq)   # (B, T, d_model)
                f = out.mean(dim=1)
            elif self.seq_type == "gru":
                _, h_n = self.sequence(z_seq)
                f = h_n[-1]
            else:
                raise ValueError(f"Unknown seq_type: {self.seq_type}")
        return f

    def forward(self, x, lambda_gr=0.0):
        f = self.forward_features(x)
        logits = self.classifier(f)

        if lambda_gr != 0.0:
            from torch.autograd import Function

            class GRL(Function):
                @staticmethod
                def forward(ctx, x): return x.clone()

                @staticmethod
                def backward(ctx, grad): return -lambda_gr * grad

            f_rev = GRL.apply(f)
        else:
            f_rev = f

        dlog = self.discriminator(f_rev)
        return logits, dlog


# ---------------- TS-TCC feature extractor ----------------
def extract_features(trainer, X, device, batch_size=256):
    """
    Use TS-TCC encoder outputs z: [B, C, T], apply GAP over time -> [B, C]
    """
    trainer.model.eval()
    feats = []

    dl = DataLoader(
        ArrayDataset(X),                 # ✅ bỏ return_label, chỉ truyền X
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for xb in dl:
            # xb là Tensor [B, T, C] do ArrayDataset trả ra
            xb = xb.permute(0, 2, 1).to(device)   # [B, C, T]
            _, z = trainer.model(xb)              # z: [B, C, T]
            f = z.mean(dim=2)                     # [B, C]
            feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)


def fit_deepsvdd_on_source(Zs_np, device, hidden_dim=128, rep_dim=64, nu=0.05,
                           epochs=20, warmup_epochs=5, lr=1e-3, bs=512, seed=42):
    """
    Train DeepSVDD (soft-boundary) on source TS-TCC features (assumed normal-ish).
    Warmup: first warmup_epochs optimize only dist2 mean (no R term) to stabilize.
    """
    torch.manual_seed(seed)
    Zs = torch.tensor(Zs_np, dtype=torch.float32, device=device)

    svdd = DeepSVDD(in_dim=Zs.shape[1], hidden_dim=hidden_dim, rep_dim=rep_dim).to(device)

    svdd.init_center(Zs)
    opt = torch.optim.Adam(svdd.parameters(), lr=lr)

    n = Zs.shape[0]
    for ep in range(epochs):
        svdd.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            zb = Zs[idx]

            if ep < warmup_epochs:
                dist2 = svdd(zb)           # [B]
                loss = dist2.mean()
            else:
                loss, _, _ = svdd.loss_soft_boundary(zb, nu=nu)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(
                f"[DeepSVDD] epoch {ep+1}/{epochs} | loss={total_loss/max(1,n_batches):.6f} "
                f"| R={float(svdd.R.detach().cpu()):.6f} | nu={nu}"
            )

    svdd.eval()
    return svdd

from copy import deepcopy
from typing import Dict, Any, List

# ---- Base architectures for paper (fixed, reproducible) ----
def get_base_arches(in_ch: int, num_classes: int):
    # You already import ArchConfig in search_space.py; reuse same dataclass here.
    # If ArchConfig isn't imported in pipeline.py, add:
    # from src.adaptnas.search_space import ArchConfig
    from src.adaptnas.search_space import ArchConfig

    bases = []

    # Base 1: CNN + GRU (light)
    bases.append((
        "Base_CNN_GRU",
        ArchConfig(
            enc_filters=[32, 64],
            enc_kernels=[5, 3],
            enc_strides=[1, 2],
            enc_dilations=[1, 1],
            enc_pool=("max", 2),
            enc_activation="relu",
            seq_type="gru",
            seq_layers=1,
            seq_heads=1,
            seq_hidden=128,
            seq_kernel=3,
            seq_dilation=1,
            clf_layers=2,
            clf_units=64,
            d_model=128,
        )
    ))

    # Base 2: CNN + TCN
    bases.append((
        "Base_CNN_TCN",
        ArchConfig(
            enc_filters=[32, 64],
            enc_kernels=[5, 3],
            enc_strides=[1, 2],
            enc_dilations=[1, 1],
            enc_pool=("max", 2),
            enc_activation="relu",
            seq_type="tcn",
            seq_layers=2,
            seq_heads=1,
            seq_hidden=128,
            seq_kernel=5,
            seq_dilation=2,
            clf_layers=2,
            clf_units=64,
            d_model=128,
        )
    ))

    # Base 3: CNN + Transformer (small)
    bases.append((
        "Base_CNN_TRF",
        ArchConfig(
            enc_filters=[32, 64],
            enc_kernels=[5, 3],
            enc_strides=[1, 2],
            enc_dilations=[1, 1],
            enc_pool=("max", 2),
            enc_activation="relu",
            seq_type="transformer",
            seq_layers=1,
            seq_heads=2,
            seq_hidden=128,
            seq_kernel=3,
            seq_dilation=1,
            clf_layers=2,
            clf_units=64,
            d_model=128,
        )
    ))

    return bases


def run_final_only_option2(
    arch_name: str,
    arch_cfg,
    Xs, Ys, Xt, Yt,
    fixed_s_idx,
    args,
    device,
    in_ch,
    num_classes,
    N_ITERS,
    out_dir="outputs",
):
    """
    FINAL TRAINING ONLY for a given arch, using Option 2 weighting:
      warmup on source -> freeze -> extract forward_features -> SVDD -> robust sigmoid weights -> train_bilevel final
    Returns dict of metrics & logs.
    """

    # ---- hyperparams (match what you used in Option2 search) ----
    svdd_epochs = 10
    svdd_warmup_epochs = 2
    svdd_nu = 0.05

    tau = 1.0
    w_min = 0.05

    # IMPORTANT: do NOT subsample target for weighting (you already noticed it can hurt)
    max_svdd_fit = 5000    # ok to subsample SOURCE for fitting
    max_svdd_score = None  # DO NOT subsample target

    # build model
    model = CandidateModel(in_ch, arch_cfg, num_classes).to(device)

    # datasets
    ds_source = ArrayDataset(Xs, Ys)
    ds_target_val = ArrayDataset(Xt, Yt)

    # val loader must reuse fixed_s_idx from search stage for fairness
    val_loader, _, _, _, _ = build_validation(
        ds_source, ds_target_val,
        bs=args.batch_size, seed=42, fixed_s_idx=fixed_s_idx
    )

    # ---- (A) Warmup on source so features are meaningful ----
    warmup_candidate_on_source(
        model, ds_source, device=device,
        steps=80, bs=args.batch_size, lr=1e-3
    )

    # ---- (B) Freeze for feature extraction + SVDD ----
    _set_requires_grad(model, False)

    if max_svdd_fit is not None and len(Xs) > max_svdd_fit:
        idx_fit = np.random.RandomState(42).choice(len(Xs), size=max_svdd_fit, replace=False)
        Xs_fit = Xs[idx_fit]
    else:
        Xs_fit = Xs

    Fs = extract_candidate_features(model, Xs_fit, device=device, batch_size=256)
    Ft = extract_candidate_features(model, Xt, device=device, batch_size=256)  # full target

    svdd = fit_deepsvdd_on_features(
        Fs, device=device,
        hidden_dim=128, rep_dim=64,
        nu=svdd_nu,
        epochs=svdd_epochs,
        warmup_epochs=svdd_warmup_epochs,
        lr=1e-3, bs=1024, seed=42
    )

    # ✅ score TOÀN BỘ Xt bằng streaming (không cần pre-extract Ft)
    raw = score_candidate_svdd_stream(
        model, svdd, Xt, device=device,
        batch_size=256, mode="dist2"
    )
    w_ent = robust_sigmoid_weights(raw, tau=tau, w_min=w_min)


    print(
        f"[FINAL-ONLY][{arch_name}] weights: "
        f"mean={w_ent.mean():.4f} min={w_ent.min():.4f} max={w_ent.max():.4f} | "
        f"tau={tau} w_min={w_min} nu={svdd_nu}"
    )

    # ---- (C) Unfreeze for training ----
    _set_requires_grad(model, True)

    # IMPORTANT: ensure training mode before backward for GRU/LSTM cuDNN
    model.train()

    ds_target = ArrayDataset(Xt, None, w=w_ent)

    alpha_final = 0.3 + 0.2 * (N_ITERS - 1)

    train_log = train_bilevel(
        model, ds_source, ds_target, val_loader,
        device=device,
        steps=200,
        bs=args.batch_size,
        alpha=alpha_final,
        gamma=1.0,
        lr_inner=1e-3,
        lr_arch=1e-3,
        use_cosine_decay=True,
        early_stop=True,
        patience=10,
        ckpt_path=os.path.join("outputs", "checkpoints", f"{arch_name}_final_best.pt")
    )

    # ---- Eval (same as your common eval block, isolated here) ----
    metrics_paper = {}
    metrics_uad = None

    if Xt is not None and Yt is not None:
        model.eval()
        all_probs = []
        dl_eval = DataLoader(ArrayDataset(Xt), batch_size=256, shuffle=False)
        with torch.no_grad():
            for xb in dl_eval:
                xb = xb.to(device)
                logits, _ = model(xb)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        pred = probs.argmax(axis=1)

        target_acc = float((pred == Yt).mean())
        target_err = 100.0 * (1.0 - target_acc)
        metrics_paper = {"target_top1_acc": target_acc, "target_error_percent": target_err}

        if probs.shape[1] == 2:
            scores = probs[:, 1]
            ap, auroc = compute_ap_auroc(Yt, scores)

            # POT threshold learned from source-normal (Xs)
            train_probs = []
            dl_train_norm = DataLoader(ArrayDataset(Xs), batch_size=256, shuffle=False)
            with torch.no_grad():
                for xb in dl_train_norm:
                    xb = xb.to(device)
                    logits, _ = model(xb)
                    train_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            train_scores = np.concatenate(train_probs, axis=0)[:, 1]

            thr_pot = pot_threshold(train_scores, q=1e-3, level=0.99)
            p_pot, r_pot, f1_pot = f1_at_threshold(Yt, scores, thr_pot)

            f1_best, p_best, r_best, thr_best = best_f1(Yt, scores)

            y_pred_bin = (scores >= thr_pot).astype(int)
            ev = event_f1_and_delay(Yt, y_pred_bin)

            metrics_uad = {
                "ap": float(ap),
                "auroc": float(auroc),
                "f1_pot": float(f1_pot),
                "precision_pot": float(p_pot),
                "recall_pot": float(r_pot),
                "thr_pot": float(thr_pot),
                "f1_best": float(f1_best),
                "precision_best": float(p_best),
                "recall_best": float(r_best),
                "thr_best": float(thr_best),
                "event_f1": float(ev["event_f1"]),
                "event_precision": float(ev["event_precision"]),
                "event_recall": float(ev["event_recall"]),
                "delay_mean": float(ev["delay_mean"]),
                "delay_median": float(ev["delay_median"]),
            }

    return {
        "arch_name": arch_name,
        "arch": str(arch_cfg),
        "metrics_paper": metrics_paper,
        "metrics_uad": metrics_uad,
        "train_curves": train_log,
        "weighting": {
            "option": "option2_svdd_on_forward_features",
            "tau": float(tau),
            "w_min": float(w_min),
            "svdd_nu": float(svdd_nu),
            "svdd_epochs": int(svdd_epochs),
            "svdd_warmup_epochs": int(svdd_warmup_epochs),
        }
    }




@torch.no_grad()
def deepsvdd_anomaly_score(svdd, Z_np, device, bs=2048, mode="dist2"):
    """
    mode:
      - "dist2": use distance^2 to center (continuous score)  ✅ recommended for weighting
      - "dist2_minus_R2": dist2 - R^2
      - "slack": relu(dist2 - R^2)
    Return np.ndarray [N], higher = more anomalous
    """
    Z = torch.tensor(Z_np, dtype=torch.float32, device=device)
    scores = []
    R2 = (svdd.R ** 2)

    n = Z.shape[0]
    for i in range(0, n, bs):
        zb = Z[i:i+bs]
        dist2 = svdd(zb)

        if mode == "dist2":
            sc = dist2
        elif mode == "dist2_minus_R2":
            sc = dist2 - R2
        else:  # "slack"
            sc = torch.relu(dist2 - R2)

        scores.append(sc.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)

def eval_auroc_on_loader(model, loader, device):
    model.eval()
    all_scores, all_y = [], []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                xb, yb = batch[0], batch[1]
            elif isinstance(batch, dict):
                xb = batch.get("x") or batch.get("input") or list(batch.values())[0]
                yb = batch.get("y") or batch.get("label") or list(batch.values())[1]
            else:
                xb, yb = batch

            xb = xb.to(device)
            yb = yb.to(device)

            logits, _ = model(xb)
            probs = torch.softmax(logits, dim=1)

            # binary anomaly: score = P(anomaly) = probs[:,1]
            # multi-class: dùng 1 - P(class0) như "anomaly-ness" (fallback)
            if probs.size(1) == 2:
                score = probs[:, 1]
            else:
                score = 1.0 - probs[:, 0]

            all_scores.append(score.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())

    if not all_scores:
        return 0.5

    scores = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    # nếu y_true không có đủ 2 class -> roc_auc_score sẽ lỗi
    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = 0.5

    return float(auroc)

import math

def _set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def extract_candidate_features(cand, X_np, device, batch_size=256):
    cand.eval()
    feats = []
    dl = DataLoader(ArrayDataset(X_np), batch_size=batch_size, shuffle=False)
    for xb in dl:
        xb = xb.to(device)
        f = cand.forward_features(xb)  # [B, D]
        feats.append(f.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

def warmup_candidate_on_source(cand, ds_source, device, steps=50, bs=64, lr=1e-3):
    cand.train()
    opt = torch.optim.Adam(cand.parameters(), lr=lr)
    dl = DataLoader(ds_source, batch_size=min(bs, len(ds_source)), shuffle=True, drop_last=False)
    it = iter(dl)

    for _ in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)

        xb = xb.to(device)
        yb = yb.to(device)

        logits, _ = cand(xb, lambda_gr=0.0)
        loss = torch.nn.functional.cross_entropy(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

def fit_deepsvdd_on_features(Zs_np, device, hidden_dim=128, rep_dim=64, nu=0.05,
                             epochs=10, warmup_epochs=2, lr=1e-3, bs=1024, seed=42):
    torch.manual_seed(seed)
    Zs = torch.tensor(Zs_np, dtype=torch.float32, device=device)

    svdd = DeepSVDD(in_dim=Zs.shape[1], hidden_dim=hidden_dim, rep_dim=rep_dim).to(device)
    svdd.init_center(Zs)

    opt = torch.optim.Adam(svdd.parameters(), lr=lr)
    n = Zs.shape[0]

    for ep in range(epochs):
        svdd.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            zb = Zs[perm[i:i+bs]]
            if ep < warmup_epochs:
                dist2 = svdd(zb)
                loss = dist2.mean()
            else:
                loss, _, _ = svdd.loss_soft_boundary(zb, nu=nu)

            opt.zero_grad()
            loss.backward()
            opt.step()

    svdd.eval()
    return svdd

def robust_sigmoid_weights(raw, tau=1.0, w_min=0.05):
    raw = np.asarray(raw)
    med = np.median(raw)
    q25 = np.percentile(raw, 25)
    q75 = np.percentile(raw, 75)
    iqr = max(1e-8, q75 - q25)
    z = (raw - med) / iqr

    w = 1.0 / (1.0 + np.exp(z / max(1e-8, tau)))
    w = np.clip(w, w_min, 1.0).astype(np.float32)
    return w

@torch.no_grad()
def score_candidate_svdd_stream(cand, svdd, X_np, device, batch_size=256, mode="dist2"):
    # NOTE: streaming score full target, tránh subsample target
    cand.eval()
    svdd.eval()
    scores = []
    R2 = (svdd.R ** 2)

    dl = DataLoader(ArrayDataset(X_np), batch_size=batch_size, shuffle=False)
    for xb in dl:
        xb = xb.to(device)
        f = cand.forward_features(xb)  # [B, D]
        dist2 = svdd(f)

        if mode == "dist2":
            sc = dist2
        elif mode == "dist2_minus_R2":
            sc = dist2 - R2
        else:
            sc = torch.relu(dist2 - R2)

        scores.append(sc.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)






# ========================= Main =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_or_paths", required=True)
    parser.add_argument("--epochs_pretrain", type=int, default=10)
    parser.add_argument("--search_candidates", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--uad",
        action="store_true",
        help="UAD mode: train_normal.npz,val_mixed.npz[,test_mixed.npz]. "
             "Train target is unlabeled val_mixed.X, validation uses val_mixed labels.",
    )
    args = parser.parse_args()

    set_global_seed(42)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    ds = args.dataset_or_paths
    device = args.device

    def load_Xy(npz_path):
        X, y = load_npz_if_exists(npz_path)
        return X, (None if y is None else y.astype(int))

    # -------- Load data --------
    if args.uad:
        parts = [p.strip() for p in ds.split(",")]
        if len(parts) < 2:
            raise ValueError("UAD mode expects: train_normal.npz,val_mixed.npz[,test_mixed.npz]")

        Xs, _ = load_Xy(parts[0])         # normal only
        Xval, Yval = load_Xy(parts[1])    # mixed + labels
        Xtest, Ytest = (None, None)
        if len(parts) >= 3:
            Xtest, Ytest = load_Xy(parts[2])

        print("UAD mode:")
        print("  train_normal:", Xs.shape)
        print("  val_mixed   :", Xval.shape, "(labels:", (Yval is not None), ")")
        if Xtest is not None:
            print("  test_mixed  :", Xtest.shape, "(labels:", (Ytest is not None), ")")

        print("[INFO] Normalizing window size to 128...")
        Xs = fix_length(Xs, window=128)
        Xval = fix_length(Xval, window=128)
        if Xtest is not None:
            Xtest = fix_length(Xtest, window=128)

        in_ch = Xs.shape[-1]
        num_classes = 2

    else:
        ds_low = ds.lower()
        X_pretrain_multi = None

        if "," in ds_low:
            s_path, t_path = ds_low.split(",", 1)
            s_path = s_path.strip()
            t_path = t_path.strip()
            Xs, Ys = load_npz_if_exists(s_path)
            Xt, Yt = load_npz_if_exists(t_path)
            Ys = Ys.astype(int)
            Yt = Yt.astype(int) if Yt is not None else None

            if "data/smd" in s_path and "machine-" in s_path:
                machine_dir = os.path.dirname(s_path)
                smd_root = os.path.dirname(machine_dir)
                X_pretrain_multi = load_all_smd_for_pretrain(smd_root, window=128)

        elif ds_low == "uci_har":
            Xs, Ys = load_npz_if_exists("data/uci_har/source.npz")
            Xt, Yt = load_npz_if_exists("data/uci_har/target.npz")
            Ys = Ys.astype(int)
            Yt = Yt.astype(int) if Yt is not None else None

        elif ds_low in ("sleepedf", "sleep_edf", "sleep-edf"):
            Xs, Ys = load_npz_if_exists("data/sleepedf/source.npz")
            Xt, Yt = load_npz_if_exists("data/sleepedf/target.npz")
            Ys = Ys.astype(int)
            Yt = Yt.astype(int) if Yt is not None else None

        else:
            raise ValueError("Unknown dataset name / format.")

        print("[INFO] Normalizing window size to 128...")
        Xs = fix_length(Xs, window=128)
        Xt = fix_length(Xt, window=128)
        in_ch = Xs.shape[-1]

        if Yt is not None:
            num_classes = int(max(Ys.max(), Yt.max())) + 1
        else:
            num_classes = int(Ys.max()) + 1
        num_classes = max(num_classes, 2)
        print(f"[INFO] Detected num_classes = {num_classes} (in_ch={in_ch})")

    # -------- Stage 1: TS-TCC pretraining --------
    from src.ts_tcc.models.model import base_Model
    from src.ts_tcc.models.TC import TC
    from src.ts_tcc.config_files.HAR_Configs import Config
    from src.ts_tcc.dataloader.dataloader import Load_Dataset

    print("[INFO] TS-TCC pretraining ...")
    config = Config()
    config.input_channels = in_ch
    if hasattr(config, "input_length"):
        config.input_length = 128
    if hasattr(config, "num_classes"):
        config.num_classes = num_classes
    config.batch_size = args.batch_size
    config.num_epoch = args.epochs_pretrain

    model = base_Model(config).to(device)
    temporal_contr_model = TC(config, device).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=3e-4)
    temp_opt = torch.optim.Adam(temporal_contr_model.parameters(), lr=config.lr, weight_decay=3e-4)
    tstcc = TSTrainer(model, temporal_contr_model, model_opt, temp_opt, device, config)

    if args.uad:
        train_x = Xs
        print(f"[INFO] TS-TCC pretraining on train_normal only: {train_x.shape[0]} windows.")
    else:
        if X_pretrain_multi is not None:
            train_x = X_pretrain_multi
            print(f"[INFO] TS-TCC pretraining on multi-machine SMD: {train_x.shape[0]} windows.")
        else:
            train_x = np.concatenate([Xs, Xt], axis=0)
            print(f"[INFO] TS-TCC pretraining on current domain only: {train_x.shape[0]} windows.")

    train_ss = {
        "samples": torch.tensor(train_x, dtype=torch.float32),
        "labels": torch.zeros(len(train_x)),
    }
    train_dataset = Load_Dataset(train_ss, config, training_mode="self_supervised")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    tstcc.train(train_dl=train_loader, training_mode="self_supervised")

    # -------- Stage 2-4: AdaptNAS iterative search --------
    N_ITERS = 3
    history = []
    fixed_s_idx = None

    if args.uad:
        # UAD: source is normal labeled 0, target is unlabeled Xval (train), validation is (Xval,Yval)
        Ys = np.zeros(len(Xs), dtype=int)
        Xtu = Xval.copy()
        Yval = Yval.astype(int) if Yval is not None else None

        ds_source = ArrayDataset(Xs, Ys)
        dl_val = DataLoader(ArrayDataset(Xval, Yval), batch_size=args.batch_size, shuffle=False)

        n_src_val = min(len(Xs), 200)
        idx_src_val = np.random.RandomState(42).choice(len(Xs), size=n_src_val, replace=False)
        val_src = DataLoader(
            ArrayDataset(Xs[idx_src_val], np.zeros(n_src_val, dtype=int)),
            batch_size=args.batch_size,
            shuffle=False,
        )
        val_tgt = dl_val

        best_arch = None
        best_cand = None

        for iter_id in range(N_ITERS):
            print(f"\n[ITER {iter_id+1}/{N_ITERS}] AdaptNAS search (UAD)...")

            ds_target_u = ArrayDataset(Xtu)  # unlabeled

            best_val_acc = -float("inf")
            best_arch_iter = None
            best_cand_iter = None

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes=2).to(device)

                alpha_iter = 0.3 + 0.2 * iter_id

                train_bilevel(
                    cand, ds_source, ds_target_u, dl_val,
                    device=device, steps=50, bs=args.batch_size,
                    alpha=alpha_iter, gamma=1.0,
                    use_cosine_decay=True, early_stop=False
                )

                from src.adaptnas.optimizer import AdaptNASOptimizer
                opt = AdaptNASOptimizer(cand, alpha=0.5, gamma=1.0, lr_inner=1e-3, lr_arch=1e-3, device=device)
                stats = opt.step_upper_combined(val_src, val_tgt, alpha=alpha_iter)
                val_acc = 1.0 - stats["hybrid_err"]

                history.append({
                    "iter": iter_id + 1,
                    "arch": str(arch_c),
                    "val_acc": val_acc,
                    "src_err": stats["src_err"],
                    "tgt_err": stats["tgt_err"],
                    "hybrid_err": stats["hybrid_err"],
                    "alpha": alpha_iter,
                })

                print(
                    f"  Candidate {i+1}/{args.search_candidates}: "
                    f"val_acc={val_acc:.4f} | src_err={stats['src_err']:.4f} | tgt_err={stats['tgt_err']:.4f}"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_arch_iter = arch_c
                    best_cand_iter = cand

            best_arch = best_arch_iter
            best_cand = best_cand_iter

            print(f"[ITER {iter_id+1}] ✅ Best arch (by val_acc) = {best_arch}")
            print("[INFO] Selected architecture summary:")
            print(best_cand)

    else:
        # ================= NON-UAD: Option 2 (SVDD on candidate forward_features) =================
        best_overall_auroc = -1.0
        best_overall_val_acc = -1.0
        best_overall_arch = None
        best_overall_iter = None
        best_overall_cand_state = None
        fixed_s_idx = None

        # Hyperparams Option 2
        svdd_epochs = 10          # nặng -> giữ vừa phải
        svdd_warmup_epochs = 2
        svdd_nu = 0.05
        tau = 1.0
        w_min = 0.05

        # dùng subset để đỡ nặng khi extract features
        max_svdd_fit = 5000       # số sample source dùng fit SVDD
        max_svdd_score = 8000     # số sample target dùng score (để weight) nếu bạn muốn giảm nặng, có thể None

        best_arch = None
        best_cand = None

        for iter_id in range(N_ITERS):
            print(f"\n[ITER {iter_id+1}/{N_ITERS}] Option2: SVDD-on-candidate-features & NAS search...")

            # 6) datasets / validation (giống trước)
            ds_source = ArrayDataset(Xs, Ys)
            ds_target_val = ArrayDataset(Xt, Yt)

            if iter_id == 0:
                val_loader, fixed_s_idx, val_src, val_tgt, _ = build_validation(
                    ds_source, ds_target_val,
                    bs=args.batch_size, seed=42, fixed_s_idx=None
                )
            else:
                val_loader, _, val_src, val_tgt, _ = build_validation(
                    ds_source, ds_target_val,
                    bs=args.batch_size, seed=42, fixed_s_idx=fixed_s_idx
                )

            best_iter_auroc = -1.0
            best_iter_val_acc = -1.0
            best_arch_iter = None
            best_cand_iter = None

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes).to(device)

                alpha_iter = 0.3 + 0.2 * iter_id

                # ---- (A) Warmup nhẹ trên source để feature space có nghĩa ----
                warmup_candidate_on_source(
                    cand, ds_source, device=device,
                    steps=50, bs=args.batch_size, lr=1e-3
                )

                # ---- (B) Freeze tạm thời để train SVDD trên feature space cand ----
                _set_requires_grad(cand, False)

                # Subsample SOURCE (được), không subsample TARGET
                if max_svdd_fit is not None and len(Xs) > max_svdd_fit:
                    idx_fit = np.random.RandomState(42).choice(len(Xs), size=max_svdd_fit, replace=False)
                    Xs_fit = Xs[idx_fit]
                else:
                    Xs_fit = Xs

                Fs = extract_candidate_features(cand, Xs_fit, device=device, batch_size=256)

                svdd = fit_deepsvdd_on_features(
                    Fs, device=device,
                    hidden_dim=128, rep_dim=64,
                    nu=svdd_nu,
                    epochs=svdd_epochs,
                    warmup_epochs=svdd_warmup_epochs,
                    lr=1e-3, bs=1024, seed=42
                )

                # ✅ score TOÀN BỘ Xt bằng streaming (KHÔNG subsample target)
                raw = score_candidate_svdd_stream(
                    cand, svdd, Xt, device=device,
                    batch_size=256, mode="dist2"
                )
                w_ent = robust_sigmoid_weights(raw, tau=tau, w_min=w_min)

                print(f"    [W] w_ent stats: mean={w_ent.mean():.4f} min={w_ent.min():.4f} max={w_ent.max():.4f}")

                # ---- (C) Unfreeze lại để train bilevel ----
                _set_requires_grad(cand, True)
                cand.train()  # ✅ FIX cuDNN GRU backward


                ds_target = ArrayDataset(Xt, None, w=w_ent)

                # ---- (D) Bilevel training ----
                train_bilevel(
                    cand, ds_source, ds_target, val_loader,
                    device=device,
                    steps=80,
                    bs=args.batch_size,
                    alpha=alpha_iter,
                    gamma=1.0,
                    lr_inner=1e-3,
                    lr_arch=1e-3,
                    use_cosine_decay=True,
                    early_stop=False
                )

                # ---- (E) Eval: val_acc (hybrid_err) + AUROC on val_tgt ----
                from src.adaptnas.optimizer import AdaptNASOptimizer
                opt = AdaptNASOptimizer(
                    cand, alpha=0.5, gamma=1.0,
                    lr_inner=1e-2, lr_arch=3e-3,
                    device=device
                )
                stats = opt.step_upper_combined(val_src, val_tgt, alpha=alpha_iter)
                val_acc = 1.0 - stats["hybrid_err"]
                auroc_tgt = eval_auroc_on_loader(cand, val_tgt, device)

                history.append({
                    "iter": iter_id + 1,
                    "arch": str(arch_c),
                    "auroc_tgt": float(auroc_tgt),
                    "val_acc": float(val_acc),
                    "src_err": float(stats["src_err"]),
                    "tgt_err": float(stats["tgt_err"]),
                    "hybrid_err": float(stats["hybrid_err"]),
                    "alpha": float(alpha_iter),
                    "tau": float(tau),
                    "w_min": float(w_min),
                    "svdd_nu": float(svdd_nu),
                })

                print(f"  Candidate {i+1}/{args.search_candidates}: AUROC={auroc_tgt:.4f} | val_acc={val_acc:.4f}")

                # best in this iter
                if (auroc_tgt > best_iter_auroc) or (auroc_tgt == best_iter_auroc and val_acc > best_iter_val_acc):
                    best_iter_auroc = auroc_tgt
                    best_iter_val_acc = val_acc
                    best_arch_iter = arch_c
                    best_cand_iter = cand

            print(f"[ITER {iter_id+1}] ✅ Best iter arch = {best_arch_iter} | AUROC={best_iter_auroc:.4f} | val_acc={best_iter_val_acc:.4f}")

            # best overall across all iters
            if (best_iter_auroc > best_overall_auroc) or (best_iter_auroc == best_overall_auroc and best_iter_val_acc > best_overall_val_acc):
                best_overall_auroc = float(best_iter_auroc)
                best_overall_val_acc = float(best_iter_val_acc)
                best_overall_arch = best_arch_iter
                best_overall_iter = iter_id + 1
                best_overall_cand_state = {k: v.detach().cpu().clone() for k, v in best_cand_iter.state_dict().items()}

        best_arch = best_overall_arch

        # rebuild best_cand (optional)
        if best_overall_cand_state is not None and best_arch is not None:
            best_cand = CandidateModel(in_ch, best_arch, num_classes).to(device)
            best_cand.load_state_dict(best_overall_cand_state)

        print(f"\n[SEARCH DONE] ✅ Best OVERALL arch = {best_arch} (iter {best_overall_iter}) | AUROC={best_overall_auroc:.4f} | val_acc={best_overall_val_acc:.4f}")


        # optional: inspect arch_params distribution
        if best_cand is not None and hasattr(best_cand, "arch_params"):
            with torch.no_grad():
                probs = torch.softmax(best_cand.arch_params, dim=0)
                print("[INFO] Architecture operation probabilities:")
                for i, p in enumerate(probs):
                    print(f"  Op {i}: {p.item():.4f}")



    # -------- Final training --------
    print("[INFO] Final training with best architecture...")

    train_log = None
    metrics, roc_data = None, None
    paper_metrics = {}
    uad_metrics = None

    if args.uad:
        final_model = CandidateModel(in_ch, best_arch, num_classes).to(device)

        Ys = np.zeros(len(Xs), dtype=int)
        ds_source = ArrayDataset(Xs, Ys)

        # target train is unlabeled Xval
        ds_target = ArrayDataset(Xval)  # unlabeled

        # validation on (Xval, Yval)
        dl_val = DataLoader(ArrayDataset(Xval, Yval), batch_size=args.batch_size, shuffle=False)

        alpha_final = 0.3 + 0.2 * (N_ITERS - 1)

        train_log = train_bilevel(
            final_model, ds_source, ds_target, dl_val,
            device=device,
            steps=400,
            bs=args.batch_size,
            alpha=alpha_final,
            gamma=1.0,
            lr_inner=1e-2,
            lr_arch=3e-3,
            use_cosine_decay=True,
            early_stop=True,
            patience=10,
            ckpt_path="outputs/checkpoints/final_best.pt"
        )

        # evaluate on test_mixed if provided
        if Xtest is not None and Ytest is not None:
            final_model.eval()
            all_probs = []
            dl_test = DataLoader(ArrayDataset(Xtest), batch_size=256, shuffle=False)
            with torch.no_grad():
                for xb in dl_test:
                    xb = xb.to(device)
                    logits, _ = final_model(xb)
                    all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs = np.concatenate(all_probs, axis=0)

            y_pred = probs.argmax(axis=1)
            from sklearn.metrics import precision_recall_fscore_support
            prec, rec, f1, _ = precision_recall_fscore_support(Ytest, y_pred, average="macro", zero_division=0)
            metrics = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

    else:
        # ================= FINAL NON-UAD: run BASE architectures + BEST NAS arch (final-only) =================
        # Runs:
        #   - Base_CNN_GRU, Base_CNN_TCN, Base_CNN_TRF (from get_base_arches)
        #   - NAS_BestArch (best_arch from search stage)
        # Saves:
        #   - outputs/baselines/{arch_name}.json (each run)
        #   - outputs/baselines_summary.json (summary + best by AUROC)
        # Loads:
        #   - outputs/checkpoints/NAS_BestArch_final_best.pt into final_model for COMMON EVAL compatibility

        os.makedirs("outputs/baselines", exist_ok=True)
        os.makedirs("outputs/checkpoints", exist_ok=True)

        # 1) Build list of architectures to run
        base_arches = get_base_arches(in_ch=in_ch, num_classes=num_classes)

        arch_list = []
        for name, arch_cfg in base_arches:
            arch_list.append((name, arch_cfg))

        # include best NAS arch found from your search stage
        arch_list.append(("NAS_BestArch", best_arch))

        results_baselines = []
        best_by_auroc = None
        best_by_auroc_val = -1.0

        for arch_name, arch_cfg in arch_list:
            print(f"\n[FINAL-ONLY] Running: {arch_name}")

            out = run_final_only_option2(
                arch_name=arch_name,
                arch_cfg=arch_cfg,
                Xs=Xs, Ys=Ys, Xt=Xt, Yt=Yt,
                fixed_s_idx=fixed_s_idx,
                args=args,
                device=device,
                in_ch=in_ch,
                num_classes=num_classes,
                N_ITERS=N_ITERS,
                out_dir="outputs",
            )

            results_baselines.append(out)

            auroc_val = -1.0
            if out.get("metrics_uad") is not None:
                auroc_val = float(out["metrics_uad"].get("auroc", -1.0))

            if auroc_val > best_by_auroc_val:
                best_by_auroc_val = auroc_val
                best_by_auroc = out

            # Save each run immediately
            with open(os.path.join("outputs", "baselines", f"{arch_name}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

        # also save summary table
        summary = {
            "best_by_auroc": best_by_auroc,
            "all": results_baselines
        }
        with open(os.path.join("outputs", "baselines_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if best_by_auroc is not None:
            print(
                f"\n[BASELINES DONE] Best by AUROC = {best_by_auroc.get('arch_name')} | "
                f"AUROC={best_by_auroc_val:.4f}"
            )
        else:
            print("\n[BASELINES DONE] No valid AUROC found (metrics_uad missing).")

        # 2) Keep a final_model for COMMON EVAL block.
        # We choose NAS_BestArch as the "main" final_model and load its checkpoint.
        final_model = CandidateModel(in_ch, best_arch, num_classes).to(device)

        ckpt = os.path.join("outputs", "checkpoints", "NAS_BestArch_final_best.pt")
        if os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu")
            # handle either raw state_dict or wrapped checkpoint
            if isinstance(sd, dict) and ("state_dict" in sd):
                sd = sd["state_dict"]
            final_model.load_state_dict(sd, strict=True)
            final_model.to(device)
            final_model.eval()
            print(f"[FINAL-ONLY] Loaded checkpoint for COMMON EVAL: {ckpt}")
        else:
            print("[WARN] NAS_BestArch checkpoint not found; COMMON EVAL may be invalid.")

        # NOTE:
        # train_log is not defined in this baseline sweep branch; keep it as None.
        train_log = None





    # -------- COMMON EVAL (paper + UAD metrics) --------
    if args.uad:
        eval_X, eval_y = Xval, Yval
    else:
        eval_X, eval_y = (Xt if "Xt" in locals() else None), (Yt if "Yt" in locals() else None)

    if eval_X is not None and eval_y is not None:
        final_model.eval()
        all_probs = []
        dl_eval = DataLoader(ArrayDataset(eval_X), batch_size=256, shuffle=False)
        with torch.no_grad():
            for xb in dl_eval:
                xb = xb.to(device)
                logits, _ = final_model(xb)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        pred = probs.argmax(axis=1)

        target_acc = float((pred == eval_y).mean())
        target_err = 100.0 * (1.0 - target_acc)
        paper_metrics = {"target_top1_acc": target_acc, "target_error_percent": target_err}

        if probs.shape[1] == 2:
            scores = probs[:, 1]
            ap, auroc = compute_ap_auroc(eval_y, scores)

            # POT threshold learned from train_normal (Xs) for UAD-style
            train_probs = []
            dl_train_norm = DataLoader(ArrayDataset(Xs), batch_size=256, shuffle=False)
            with torch.no_grad():
                for xb in dl_train_norm:
                    xb = xb.to(device)
                    logits, _ = final_model(xb)
                    train_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            train_scores = np.concatenate(train_probs, axis=0)[:, 1]

            thr_pot = pot_threshold(train_scores, q=1e-3, level=0.99)
            p_pot, r_pot, f1_pot = f1_at_threshold(eval_y, scores, thr_pot)

            f1_best, p_best, r_best, thr_best = best_f1(eval_y, scores)

            y_pred_bin = (scores >= thr_pot).astype(int)
            ev = event_f1_and_delay(eval_y, y_pred_bin)

            uad_metrics = {
                "ap": float(ap),
                "auroc": float(auroc),
                "f1_pot": float(f1_pot),
                "precision_pot": float(p_pot),
                "recall_pot": float(r_pot),
                "thr_pot": float(thr_pot),
                "f1_best": float(f1_best),
                "precision_best": float(p_best),
                "recall_best": float(r_best),
                "thr_best": float(thr_best),
                "event_f1": float(ev["event_f1"]),
                "event_precision": float(ev["event_precision"]),
                "event_recall": float(ev["event_recall"]),
                "delay_mean": float(ev["delay_mean"]),
                "delay_median": float(ev["delay_median"]),
            }

    # -------- Save results --------
    res = {"best_arch": str(best_arch), "search_history": history}
    if paper_metrics:
        res["metrics_paper"] = paper_metrics
    if uad_metrics is not None:
        res["metrics_uad"] = uad_metrics
    if metrics is not None:
        res["metrics"] = metrics
    if train_log is not None:
        res["train_curves"] = train_log

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
