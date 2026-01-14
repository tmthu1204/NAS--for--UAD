"""
Top-level orchestrator (UAD project):

- TS-TCC pretrain
- (Mode A) adaptnas_combined:
    * input: train_normal.npz, val_mixed.npz[, test_mixed.npz]
    * SVDD-weighting on TARGET (val_mixed) using SVDD fitted on SOURCE normal (train_normal)
    * bilevel AdaptNAS search (kept as your current logic: warmup -> SVDD weights -> train_bilevel -> upper step)
    * final-only baselines (Base_* + NAS_BestArch) with same weighting
    * report UAD metrics on test_mixed if provided else on val_mixed

- (Mode B) uad_source:
    * input: train_normal.npz, val_mixed.npz[, test_mixed.npz]
    * search: choose arch by SVDD objective on source normal only
        - objective = mean(dist2) on held-out normal (dist2 on candidate forward_features)
        - SVDD training uses soft-boundary with warmup epochs (same as combined style: dist2)
    * final: fit SVDD on full train_normal, score val/test by dist2, report AUROC/AP/event-F1/POT

Notes:
- For UAD metrics we assume labels are binary {0,1}. If labels are multiclass, we binarize by y>0 -> 1.
- In this UAD project we set num_classes=2 for CandidateModel.
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
from src.adaptnas.trainer import train_bilevel

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
    data = np.load(path, allow_pickle=True)
    if "X" not in data:
        raise ValueError(f"{path} missing key 'X'")
    X = data["X"]
    y = data["y"] if "y" in data else None
    return X, y


def binarize_y(y):
    if y is None:
        return None
    y = np.asarray(y).astype(int)
    return (y > 0).astype(int)


def fix_length(X, window=128):
    X_fixed = []
    for x in X:
        x = np.asarray(x)
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        T, C = x.shape
        if T < window:
            pad = np.zeros((window - T, C), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        elif T > window:
            x = x[:window]
        X_fixed.append(x)
    return np.stack(X_fixed, axis=0)


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
                blocks.append(TCNBlock(in_c, arch.d_model, kernel_size=arch.seq_kernel, dilation=dilation))
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
            h = self.sequence(z)     # (B, d_model, T)
            f = h.mean(dim=2)        # (B, d_model)
        else:
            z_seq = z.transpose(1, 2)  # (B, T, d_model)
            if self.seq_type == "transformer":
                out = self.sequence(z_seq)  # (B, T, d_model)
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
    dl = DataLoader(ArrayDataset(X), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in dl:
            xb = xb.permute(0, 2, 1).to(device)   # [B, C, T]
            _, z = trainer.model(xb)              # z: [B, C, T]
            f = z.mean(dim=2)                     # [B, C]
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------- Common helpers ----------------
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
    """
    Note: In UAD we only have normal labels (0). This warmup mainly stabilizes feature extraction.
    """
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


def fit_deepsvdd_on_features(
    Zs_np, device,
    hidden_dim=128, rep_dim=64, nu=0.05,
    epochs=10, warmup_epochs=2, lr=1e-3, bs=1024, seed=42
):
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


def eval_auroc_on_loader_binary(model, loader, device):
    model.eval()
    all_scores, all_y = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            probs = torch.softmax(logits, dim=1)
            score = probs[:, 1]
            all_scores.append(score.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())

    if not all_scores:
        return 0.5

    scores = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = 0.5

    return float(auroc)


# ---- Base architectures for final-only baselines (kept) ----
def get_base_arches(in_ch: int):
    from src.adaptnas.search_space import ArchConfig
    bases = []

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
    Xs, Ys, Xt_train, Yt_train,
    X_eval, Y_eval,
    fixed_s_idx,
    args,
    device,
    in_ch,
    N_ITERS,
    out_dir="outputs",
):
    """
    FINAL-ONLY for combined mode:
      warmup on source -> freeze -> SVDD on forward_features (fit on source normal)
      -> weights on target-train (Xt_train) -> train_bilevel final
      -> eval on X_eval/Y_eval (test if provided else val)
    """
    svdd_epochs = 10
    svdd_warmup_epochs = 2
    svdd_nu = 0.05
    tau = 1.0
    w_min = 0.05

    model = CandidateModel(in_ch, arch_cfg, num_classes=2).to(device)

    ds_source = ArrayDataset(Xs, Ys)
    ds_target_val = ArrayDataset(Xt_train, Yt_train)

    val_loader, _, _, _, _ = build_validation(
        ds_source, ds_target_val,
        bs=args.batch_size, seed=42, fixed_s_idx=fixed_s_idx
    )

    warmup_candidate_on_source(model, ds_source, device=device, steps=80, bs=args.batch_size, lr=1e-3)

    _set_requires_grad(model, False)
    if len(Xs) > 5000:
        idx_fit = np.random.RandomState(42).choice(len(Xs), size=5000, replace=False)
        Xs_fit = Xs[idx_fit]
    else:
        Xs_fit = Xs

    Fs = extract_candidate_features(model, Xs_fit, device=device, batch_size=256)
    svdd = fit_deepsvdd_on_features(
        Fs, device=device, hidden_dim=128, rep_dim=64, nu=svdd_nu,
        epochs=svdd_epochs, warmup_epochs=svdd_warmup_epochs, lr=1e-3, bs=1024, seed=42
    )

    raw = score_candidate_svdd_stream(model, svdd, Xt_train, device=device, batch_size=256, mode="dist2")
    w_ent = robust_sigmoid_weights(raw, tau=tau, w_min=w_min)

    print(
        f"[FINAL-ONLY][{arch_name}] weights: mean={w_ent.mean():.4f} "
        f"min={w_ent.min():.4f} max={w_ent.max():.4f} | tau={tau} w_min={w_min} nu={svdd_nu}"
    )

    _set_requires_grad(model, True)
    model.train()

    ds_target = ArrayDataset(Xt_train, None, w=w_ent)
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
        ckpt_path=os.path.join(out_dir, "checkpoints", f"{arch_name}_final_best.pt")
    )

    metrics_uad = None
    if X_eval is not None and Y_eval is not None:
        model.eval()
        all_probs = []
        dl_eval = DataLoader(ArrayDataset(X_eval), batch_size=256, shuffle=False)
        with torch.no_grad():
            for xb in dl_eval:
                xb = xb.to(device)
                logits, _ = model(xb)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)

        scores = probs[:, 1]
        ap, auroc = compute_ap_auroc(Y_eval, scores)

        # POT threshold learned from source normal (Xs) via model's anomaly prob on Xs
        train_probs = []
        dl_train_norm = DataLoader(ArrayDataset(Xs), batch_size=256, shuffle=False)
        with torch.no_grad():
            for xb in dl_train_norm:
                xb = xb.to(device)
                logits, _ = model(xb)
                train_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        train_scores = np.concatenate(train_probs, axis=0)[:, 1]

        thr_pot = pot_threshold(train_scores, q=1e-3, level=0.99)
        p_pot, r_pot, f1_pot = f1_at_threshold(Y_eval, scores, thr_pot)

        f1_b, p_b, r_b, thr_b = best_f1(Y_eval, scores)

        y_pred_bin = (scores >= thr_pot).astype(int)
        ev = event_f1_and_delay(Y_eval, y_pred_bin)

        metrics_uad = {
            "ap": float(ap),
            "auroc": float(auroc),
            "f1_pot": float(f1_pot),
            "precision_pot": float(p_pot),
            "recall_pot": float(r_pot),
            "thr_pot": float(thr_pot),
            "f1_best": float(f1_b),
            "precision_best": float(p_b),
            "recall_best": float(r_b),
            "thr_best": float(thr_b),
            "event_f1": float(ev["event_f1"]),
            "event_precision": float(ev["event_precision"]),
            "event_recall": float(ev["event_recall"]),
            "delay_mean": float(ev["delay_mean"]),
            "delay_median": float(ev["delay_median"]),
        }

    return {
        "arch_name": arch_name,
        "arch": str(arch_cfg),
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


# ---------------- UAD-SOURCE: SVDD objective search (dist2 only) ----------------
def svdd_objective_on_source_normal(
    cand,
    X_train_norm,
    X_val_norm,
    device,
    svdd_epochs=10,
    svdd_warmup_epochs=2,
    svdd_nu=0.05,
    max_fit=5000,
):
    """
    Objective to MINIMIZE:
      - fit SVDD on train normal features (candidate forward_features)
      - compute mean(dist2) on val normal
    This matches "dist2 on candidate features" style used in combined.
    """
    _set_requires_grad(cand, False)
    cand.eval()

    if max_fit is not None and len(X_train_norm) > max_fit:
        idx_fit = np.random.RandomState(42).choice(len(X_train_norm), size=max_fit, replace=False)
        X_fit = X_train_norm[idx_fit]
    else:
        X_fit = X_train_norm

    F_fit = extract_candidate_features(cand, X_fit, device=device, batch_size=256)
    svdd = fit_deepsvdd_on_features(
        F_fit, device=device,
        hidden_dim=128, rep_dim=64,
        nu=svdd_nu,
        epochs=svdd_epochs,
        warmup_epochs=svdd_warmup_epochs,
        lr=1e-3, bs=1024, seed=42
    )

    F_val = extract_candidate_features(cand, X_val_norm, device=device, batch_size=256)
    with torch.no_grad():
        dist2 = svdd(torch.tensor(F_val, dtype=torch.float32, device=device)).detach().cpu().numpy()

    obj = float(dist2.mean())
    info = {"mean_dist2": float(dist2.mean())}
    return obj, info


def fit_final_svdd_and_score(
    cand,
    X_train_norm,
    X_eval,
    device,
    svdd_epochs=20,
    svdd_warmup_epochs=5,
    svdd_nu=0.05,
    max_fit=5000,
):
    """
    Fit SVDD on full train normal-only features, then score eval data by dist2.
    Returns: scores_eval (np.ndarray), scores_train_norm (np.ndarray)
    """
    cand.eval()
    _set_requires_grad(cand, False)

    if max_fit is not None and len(X_train_norm) > max_fit:
        idx_fit = np.random.RandomState(42).choice(len(X_train_norm), size=max_fit, replace=False)
        X_fit = X_train_norm[idx_fit]
    else:
        X_fit = X_train_norm

    F_fit = extract_candidate_features(cand, X_fit, device=device, batch_size=256)
    svdd = fit_deepsvdd_on_features(
        F_fit, device=device,
        hidden_dim=128, rep_dim=64,
        nu=svdd_nu,
        epochs=svdd_epochs,
        warmup_epochs=svdd_warmup_epochs,
        lr=1e-3, bs=1024, seed=42
    )

    scores_train = score_candidate_svdd_stream(cand, svdd, X_train_norm, device=device, batch_size=256, mode="dist2")
    scores_eval = score_candidate_svdd_stream(cand, svdd, X_eval, device=device, batch_size=256, mode="dist2")
    return scores_eval, scores_train


# ========================= Main =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_or_paths", required=True,
                        help="UAD format: train_normal.npz,val_mixed.npz[,test_mixed.npz]")
    parser.add_argument("--epochs_pretrain", type=int, default=10)
    parser.add_argument("--search_candidates", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mode",
        type=str,
        default="adaptnas_combined",
        choices=["adaptnas_combined", "uad_source"],
        help=(
            "adaptnas_combined: UAD input, SVDD-weighted target + bilevel AdaptNAS + final-only baselines.\n"
            "uad_source: UAD input, source-only (train_normal) SVDD objective NAS; final DeepSVDD scoring."
        ),
    )
    args = parser.parse_args()

    set_global_seed(42)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    device = args.device

    def load_Xy(npz_path):
        X, y = load_npz_if_exists(npz_path)
        y = None if y is None else binarize_y(y)
        return X, y

    # -------- Load data (both modes use same UAD inputs) --------
    parts = [p.strip() for p in args.dataset_or_paths.split(",")]
    if len(parts) < 2:
        raise ValueError("Expect: train_normal.npz,val_mixed.npz[,test_mixed.npz]")

    X_train_norm, y_train_norm = load_Xy(parts[0])   # should be normal-only; y not required
    X_val, y_val = load_Xy(parts[1])                 # mixed; y required for metrics & (combined) upper step
    X_test, y_test = (None, None)
    if len(parts) >= 3:
        X_test, y_test = load_Xy(parts[2])

    if y_val is None:
        raise ValueError("val_mixed.npz must contain y for AUROC/AP/event-F1 evaluation.")

    print("[UAD INPUT]")
    print("  train_normal:", X_train_norm.shape, "y:", ("yes" if y_train_norm is not None else "no"))
    print("  val_mixed   :", X_val.shape, "y:", ("yes" if y_val is not None else "no"))
    if X_test is not None:
        print("  test_mixed  :", X_test.shape, "y:", ("yes" if y_test is not None else "no"))

    print("[INFO] Normalizing window size to 128...")
    X_train_norm = fix_length(X_train_norm, window=128)
    X_val = fix_length(X_val, window=128)
    if X_test is not None:
        X_test = fix_length(X_test, window=128)

    in_ch = X_train_norm.shape[-1]
    num_classes = 2  # UAD project

    # create "source labels" for combined mode: all zeros (normal)
    Ys_source = np.zeros(len(X_train_norm), dtype=int)

    # choose evaluation split for combined final report
    X_eval = X_test if (X_test is not None and y_test is not None) else X_val
    y_eval = y_test if (X_test is not None and y_test is not None) else y_val
    eval_name = "test_mixed" if (X_test is not None and y_test is not None) else "val_mixed"

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

    # pretrain data:
    # - combined: if SMD machine path detected, can pretrain multi-machine; else pretrain on train_normal + val_mixed
    # - uad_source: pretrain on train_normal only (as you wanted)
    X_pretrain_multi = None
    if "data/smd" in parts[0].replace("\\", "/") and "machine-" in parts[0].replace("\\", "/"):
        machine_dir = os.path.dirname(parts[0])
        smd_root = os.path.dirname(machine_dir)
        if os.path.isdir(smd_root):
            X_pretrain_multi = load_all_smd_for_pretrain(smd_root, window=128)

    if args.mode == "uad_source":
        train_x = X_train_norm
        print(f"[INFO] TS-TCC pretraining on train_normal only: {train_x.shape[0]} windows.")
    else:
        if X_pretrain_multi is not None:
            train_x = X_pretrain_multi
            print(f"[INFO] TS-TCC pretraining on multi-machine SMD: {train_x.shape[0]} windows.")
        else:
            train_x = np.concatenate([X_train_norm, X_val], axis=0)
            print(f"[INFO] TS-TCC pretraining on train_normal + val_mixed: {train_x.shape[0]} windows.")

    train_ss = {
        "samples": torch.tensor(train_x, dtype=torch.float32),
        "labels": torch.zeros(len(train_x)),
    }
    train_dataset = Load_Dataset(train_ss, config, training_mode="self_supervised")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    tstcc.train(train_dl=train_loader, training_mode="self_supervised")

    # -------- Stage 2-4: Search --------
    N_ITERS = 3
    history = []
    fixed_s_idx = None
    best_arch = None

    if args.mode == "uad_source":
        # split train_normal into train/val-normal for SVDD objective selection
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(X_train_norm))
        n_val = max(50, int(0.2 * len(X_train_norm))) if len(X_train_norm) >= 250 else max(1, int(0.2 * len(X_train_norm)))
        idx_val = perm[:n_val]
        idx_tr = perm[n_val:] if n_val < len(X_train_norm) else perm[:max(1, len(X_train_norm)//2)]

        Xs_tr = X_train_norm[idx_tr]
        Xs_val = X_train_norm[idx_val]

        best_obj = float("inf")
        best_state = None

        svdd_epochs = 10
        svdd_warmup_epochs = 2
        svdd_nu = 0.05

        for iter_id in range(N_ITERS):
            print(f"\n[ITER {iter_id+1}/{N_ITERS}] UAD_SOURCE search: SVDD objective on train_normal...")

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes=2).to(device)

                obj, info = svdd_objective_on_source_normal(
                    cand,
                    X_train_norm=Xs_tr,
                    X_val_norm=Xs_val,
                    device=device,
                    svdd_epochs=svdd_epochs,
                    svdd_warmup_epochs=svdd_warmup_epochs,
                    svdd_nu=svdd_nu,
                    max_fit=5000,
                )

                history.append({
                    "iter": iter_id + 1,
                    "arch": str(arch_c),
                    "svdd_obj": float(obj),
                    "svdd_mean_dist2": float(info["mean_dist2"]),
                })

                print(f"  Candidate {i+1}/{args.search_candidates}: obj(mean_dist2)={obj:.6f}")

                if obj < best_obj:
                    best_obj = float(obj)
                    best_arch = arch_c
                    best_state = {k: v.detach().cpu().clone() for k, v in cand.state_dict().items()}

        print(f"\n[UAD_SOURCE SEARCH DONE] ✅ Best arch = {best_arch} | best_obj={best_obj:.6f}")

        # rebuild best model (optional)
        if best_arch is None or best_state is None:
            raise RuntimeError("uad_source: best_arch/best_state is None after search.")
        best_cand = CandidateModel(in_ch, best_arch, num_classes=2).to(device)
        best_cand.load_state_dict(best_state)

    else:
        # ================= ADAPTNAS-COMBINED (logic kept; input adapted to UAD 3-file) =================
        Xs = X_train_norm
        Ys = Ys_source
        Xt = X_val
        Yt = y_val

        best_overall_auroc = -1.0
        best_overall_val_acc = -1.0
        best_overall_arch = None
        best_overall_iter = None
        best_overall_cand_state = None
        fixed_s_idx = None

        svdd_epochs = 10
        svdd_warmup_epochs = 2
        svdd_nu = 0.05
        tau = 1.0
        w_min = 0.05
        max_svdd_fit = 5000

        for iter_id in range(N_ITERS):
            print(f"\n[ITER {iter_id+1}/{N_ITERS}] ADAPTNAS_COMBINED: SVDD-weighting + bilevel search...")

            ds_source = ArrayDataset(Xs, Ys)
            ds_target_val = ArrayDataset(Xt, Yt)

            if iter_id == 0:
                val_loader, fixed_s_idx, val_src, val_tgt, _ = build_validation(
                    ds_source, ds_target_val, bs=args.batch_size, seed=42, fixed_s_idx=None
                )
            else:
                val_loader, _, val_src, val_tgt, _ = build_validation(
                    ds_source, ds_target_val, bs=args.batch_size, seed=42, fixed_s_idx=fixed_s_idx
                )

            best_iter_auroc = -1.0
            best_iter_val_acc = -1.0
            best_arch_iter = None
            best_cand_iter = None

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes=2).to(device)

                alpha_iter = 0.3 + 0.2 * iter_id

                # warmup on source normal-only labels (0) to stabilize feature extraction
                warmup_candidate_on_source(cand, ds_source, device=device, steps=50, bs=args.batch_size, lr=1e-3)

                # fit SVDD on candidate features from source normal
                _set_requires_grad(cand, False)

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

                raw = score_candidate_svdd_stream(cand, svdd, Xt, device=device, batch_size=256, mode="dist2")
                w_ent = robust_sigmoid_weights(raw, tau=tau, w_min=w_min)

                print(f"    [W] w_ent stats: mean={w_ent.mean():.4f} min={w_ent.min():.4f} max={w_ent.max():.4f}")

                # bilevel training
                _set_requires_grad(cand, True)
                cand.train()

                ds_target = ArrayDataset(Xt, None, w=w_ent)

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

                # upper step (uses labels on val_tgt internally)
                from src.adaptnas.optimizer import AdaptNASOptimizer
                opt = AdaptNASOptimizer(
                    cand, alpha=0.5, gamma=1.0,
                    lr_inner=1e-2, lr_arch=3e-3,
                    device=device
                )
                stats = opt.step_upper_combined(val_src, val_tgt, alpha=alpha_iter)
                val_acc = 1.0 - stats["hybrid_err"]

                auroc_tgt = eval_auroc_on_loader_binary(cand, val_tgt, device)

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

                better = (auroc_tgt > best_iter_auroc) or (auroc_tgt == best_iter_auroc and val_acc > best_iter_val_acc)
                if better:
                    best_iter_auroc = float(auroc_tgt)
                    best_iter_val_acc = float(val_acc)
                    best_arch_iter = arch_c
                    best_cand_iter = cand

            print(f"[ITER {iter_id+1}] ✅ Best iter arch = {best_arch_iter} | AUROC={best_iter_auroc:.4f} | val_acc={best_iter_val_acc:.4f}")

            better_overall = (best_iter_auroc > best_overall_auroc) or (best_iter_auroc == best_overall_auroc and best_iter_val_acc > best_overall_val_acc)
            if better_overall:
                best_overall_auroc = float(best_iter_auroc)
                best_overall_val_acc = float(best_iter_val_acc)
                best_overall_arch = best_arch_iter
                best_overall_iter = iter_id + 1
                best_overall_cand_state = {k: v.detach().cpu().clone() for k, v in best_cand_iter.state_dict().items()}

        best_arch = best_overall_arch
        if best_arch is None or best_overall_cand_state is None:
            raise RuntimeError("adaptnas_combined: best_arch/state is None after search.")

        best_cand = CandidateModel(in_ch, best_arch, num_classes=2).to(device)
        best_cand.load_state_dict(best_overall_cand_state)

        print(f"\n[SEARCH DONE] ✅ Best OVERALL arch = {best_arch} (iter {best_overall_iter}) | best_AUROC={best_overall_auroc:.4f}")

    # -------- Final stage --------
    print("[INFO] Final stage ...")

    res = {
        "mode": args.mode,
        "best_arch": str(best_arch),
        "search_history": history,
        "eval_split": eval_name,
    }

    if args.mode == "uad_source":
        final_model = CandidateModel(in_ch, best_arch, num_classes=2).to(device)

        # final: fit SVDD on full train_normal; score eval split
        scores_eval, scores_train = fit_final_svdd_and_score(
            final_model,
            X_train_norm=X_train_norm,
            X_eval=X_eval,
            device=device,
            svdd_epochs=20,
            svdd_warmup_epochs=5,
            svdd_nu=0.05,
            max_fit=5000,
        )

        ap, auroc = compute_ap_auroc(y_eval, scores_eval)

        thr_pot = pot_threshold(scores_train, q=1e-3, level=0.99)
        p_pot, r_pot, f1_pot = f1_at_threshold(y_eval, scores_eval, thr_pot)

        f1_b, p_b, r_b, thr_b = best_f1(y_eval, scores_eval)

        y_pred_bin = (scores_eval >= thr_pot).astype(int)
        ev = event_f1_and_delay(y_eval, y_pred_bin)

        res["metrics_uad"] = {
            "ap": float(ap),
            "auroc": float(auroc),
            "f1_pot": float(f1_pot),
            "precision_pot": float(p_pot),
            "recall_pot": float(r_pot),
            "thr_pot": float(thr_pot),
            "f1_best": float(f1_b),
            "precision_best": float(p_b),
            "recall_best": float(r_b),
            "thr_best": float(thr_b),
            "event_f1": float(ev["event_f1"]),
            "event_precision": float(ev["event_precision"]),
            "event_recall": float(ev["event_recall"]),
            "delay_mean": float(ev["delay_mean"]),
            "delay_median": float(ev["delay_median"]),
        }

    else:
        # combined final-only baselines (Base_* + NAS_BestArch)
        Xs = X_train_norm
        Ys = Ys_source
        Xt_train = X_val
        Yt_train = y_val

        os.makedirs("outputs/baselines", exist_ok=True)
        os.makedirs("outputs/checkpoints", exist_ok=True)

        base_arches = get_base_arches(in_ch=in_ch)
        arch_list = [(name, cfg) for name, cfg in base_arches]
        arch_list.append(("NAS_BestArch", best_arch))

        results_baselines = []
        best_by_auroc = None
        best_by_auroc_val = -1.0

        for arch_name, arch_cfg in arch_list:
            print(f"\n[FINAL-ONLY] Running: {arch_name} (eval on {eval_name})")

            out = run_final_only_option2(
                arch_name=arch_name,
                arch_cfg=arch_cfg,
                Xs=Xs, Ys=Ys,
                Xt_train=Xt_train, Yt_train=Yt_train,
                X_eval=X_eval, Y_eval=y_eval,
                fixed_s_idx=fixed_s_idx,
                args=args,
                device=device,
                in_ch=in_ch,
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

            with open(os.path.join("outputs", "baselines", f"{arch_name}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

        summary = {"best_by_auroc": best_by_auroc, "all": results_baselines}
        with open(os.path.join("outputs", "baselines_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        res["baselines_summary"] = {"best_by_auroc": best_by_auroc}
        if best_by_auroc is not None and best_by_auroc.get("metrics_uad") is not None:
            res["metrics_uad"] = best_by_auroc["metrics_uad"]

    # -------- Save results --------
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("\n[OK] Saved outputs/results.json")


if __name__ == "__main__":
    main()
