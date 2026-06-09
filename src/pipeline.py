"""
Top-level orchestrator (UAD project):

- TS-TCC pretrain
- (Mode A) adaptnas_combined:
    * input: train_normal.npz, target_pool_unlabeled.npz, val_mixed.npz, test_mixed.npz
    * one-class weighting on TARGET POOL using a backend fitted on SOURCE normal
    * bilevel AdaptNAS search with:
        - lower level on source normal + unlabeled weighted target pool
        - upper level on source holdout normal + unlabeled weighted target pool
    * final-only baselines (Base_* + NAS_BestArch) with same weighting
    * report UAD metrics on test_mixed if provided else on val_mixed

- (Mode B) uad_source:
    * input: train_normal.npz, val_mixed.npz[, test_mixed.npz]
    * search: choose arch by one-class objective on source normal only
        - objective = mean(one-class score) on held-out normal features
    * final: fit the selected one-class backend on full train_normal features,
      score val/test, and report AUROC/AP/event-F1/POT

Notes:
- For UAD metrics we assume labels are binary {0,1}. If labels are multiclass, we binarize by y>0 -> 1.
- In this UAD project we set num_classes=2 for CandidateModel.
- TS-TCC is used to initialize candidate encoders before search/final training.
- The default backend is `deepsvdd`; `autoencoder`, `knn_distance`, `oneclass_svm`,
  `svdd`, `prototype_oneclass`, and `mahalanobis_head` are also supported.
"""

import argparse
import os
import json
import random
from dataclasses import replace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import ArrayDataset
from src.data.omni_smd import (
    RawSMDMachine,
    aligned_last_point_labels,
    contiguous_train_valid_split,
)
from src.data.swat import RawSWaTDataset
from src.data.swat import build_upstream_usad_flat_windows, build_upstream_usad_window_labels
from src.data.tranad_smd import build_tranad_windows, load_raw_tranad_smd_machine
from src.ts_tcc.trainer.trainer import TSTrainer
from src.adaptnas.search_space import sample_arch
from src.adaptnas.trainer import train_bilevel
from src.families.omni_anomaly import (
    OmniAnomalyModel,
    get_fixed_paper_omni_arch,
    sample_omni_arch,
    train_omni_source,
    validate_omni_on_series,
    score_omni_series,
)
from src.families.usad import (
    UsadModel,
    get_fixed_paper_usad_arch,
    sample_usad_arch,
    train_usad_source,
    validate_usad_on_windows,
    score_usad_windows,
)
from src.families.tranad import (
    TranADModel,
    get_fixed_paper_tranad_arch,
    sample_tranad_arch,
    train_tranad_source,
    validate_tranad_on_windows,
    score_tranad_windows,
)
from src.families.omni_eval import bf_search as omni_bf_search, pot_eval as omni_pot_eval

from src.models.tscnn import EncoderCNN
from src.models.transformer import ARTransformer
from src.models.classifier import MLP
from src.models.discriminator import DomainDiscriminator
from src.oneclass import (
    OneClassConfig,
    build_oneclass_backend,
    get_oneclass_score_name,
    list_oneclass_methods,
)
from src.utils.data_paths import resolve_raw_smd_root

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


def npz_has_y(path):
    if not os.path.exists(path):
        return False
    data = np.load(path, allow_pickle=True)
    return "y" in data


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


def load_all_cached_entities_for_pretrain(root_dir="data/smd", window=128):
    import glob
    all_X = []
    n_entities = 0

    for mdir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(mdir):
            continue
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
            n_entities += 1

    if not all_X:
        raise RuntimeError(f"No cached source/target npz found under {root_dir}")
    X_all = np.concatenate(all_X, axis=0)
    print(f"[INFO] Multi-entity TS-TCC pretrain: {X_all.shape[0]} windows from {n_entities} entities.")
    return X_all


def infer_cached_pretrain_root(npz_path: str):
    machine_dir = os.path.dirname(npz_path)
    parent_dir = os.path.dirname(machine_dir)
    parent_name = os.path.basename(parent_dir).lower()

    if parent_name in {"smd", "smap", "msl"}:
        return parent_dir

    if parent_name.startswith("temporal_") or parent_name.startswith("cross_"):
        experiments_root = os.path.dirname(parent_dir)
        experiments_name = os.path.basename(experiments_root).lower()
        suffix = "_experiments"
        if experiments_name.endswith(suffix):
            dataset_name = experiments_name[: -len(suffix)]
            if dataset_name:
                return os.path.join(os.path.dirname(experiments_root), dataset_name)

    return None


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


def split_source_holdout_normal(X_source, holdout_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_source))
    n_hold = max(1, int(round(len(X_source) * holdout_ratio)))
    idx_hold = perm[:n_hold]
    idx_train = perm[n_hold:] if n_hold < len(X_source) else perm[:max(1, len(X_source) // 2)]
    return X_source[idx_train], X_source[idx_hold]


def build_upper_unlabeled_loaders(X_source_holdout, X_target_pool, w_target, bs=64):
    ds_src = ArrayDataset(X_source_holdout)
    ds_tgt = ArrayDataset(X_target_pool, None, w=w_target)
    src_loader = DataLoader(ds_src, batch_size=min(bs, max(1, len(ds_src))), shuffle=True, drop_last=False)
    tgt_loader = DataLoader(ds_tgt, batch_size=min(bs, max(1, len(ds_tgt))), shuffle=True, drop_last=False)
    return src_loader, tgt_loader


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
        self.arch = arch
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

        self.depth_projs = nn.ModuleList(
            [nn.Conv1d(out_ch, arch.d_model, kernel_size=1) for out_ch in arch.enc_filters]
        )
        self.gru_out_proj = None

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
                hidden_size=arch.seq_hidden,
                num_layers=arch.seq_layers,
                batch_first=True,
                bidirectional=False,
            )
            if arch.seq_hidden != arch.d_model:
                self.gru_out_proj = nn.Linear(arch.seq_hidden, arch.d_model)
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

        self.arch_params = nn.Parameter(torch.zeros(len(arch.enc_filters)))

    def _sequence_forward(self, z):
        if self.seq_type == "tcn":
            h = self.sequence(z)
            return h.mean(dim=2)

        z_seq = z.transpose(1, 2)
        if self.seq_type == "transformer":
            out = self.sequence(z_seq)
            return out.mean(dim=1)

        if self.seq_type == "gru":
            out, _ = self.sequence(z_seq)
            if self.gru_out_proj is not None:
                out = self.gru_out_proj(out)
            return out.mean(dim=1)

        raise ValueError(f"Unknown seq_type: {self.seq_type}")

    def forward_features(self, x):
        enc_outs = self.encoder(x, return_all=True)
        if not enc_outs:
            raise RuntimeError("EncoderCNN returned no intermediate outputs.")

        depth_logits = self.arch_params[:len(enc_outs)]
        depth_w = torch.softmax(depth_logits, dim=0)

        depth_feats = []
        for w, enc_out, proj in zip(depth_w, enc_outs, self.depth_projs):
            z = proj(enc_out.transpose(1, 2))
            depth_feats.append(w * self._sequence_forward(z))

        return torch.stack(depth_feats, dim=0).sum(dim=0)

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
def _copy_conv1d_partial(dst_conv, src_conv):
    dst_w = dst_conv.weight.data
    src_w = src_conv.weight.data.to(dst_w.device)
    new_w = torch.zeros_like(dst_w)

    out_ch = min(dst_w.size(0), src_w.size(0))
    in_ch = min(dst_w.size(1), src_w.size(1))
    k_dst = dst_w.size(2)
    k_src = src_w.size(2)

    if k_src >= k_dst:
        src_start = (k_src - k_dst) // 2
        new_w[:out_ch, :in_ch, :] = src_w[:out_ch, :in_ch, src_start:src_start + k_dst]
    else:
        dst_start = (k_dst - k_src) // 2
        new_w[:out_ch, :in_ch, dst_start:dst_start + k_src] = src_w[:out_ch, :in_ch, :]

    dst_conv.weight.copy_(new_w)

    if dst_conv.bias is not None:
        new_b = torch.zeros_like(dst_conv.bias.data)
        if src_conv.bias is not None:
            new_b[:out_ch] = src_conv.bias.data.to(new_b.device)[:out_ch]
        dst_conv.bias.copy_(new_b)


@torch.no_grad()
def _copy_bn1d_partial(dst_bn, src_bn):
    n = min(dst_bn.num_features, src_bn.num_features)
    dst_bn.weight.data[:n] = src_bn.weight.data.to(dst_bn.weight.device)[:n]
    dst_bn.bias.data[:n] = src_bn.bias.data.to(dst_bn.bias.device)[:n]
    dst_bn.running_mean.data[:n] = src_bn.running_mean.data.to(dst_bn.running_mean.device)[:n]
    dst_bn.running_var.data[:n] = src_bn.running_var.data.to(dst_bn.running_var.device)[:n]


@torch.no_grad()
def initialize_candidate_from_tstcc(cand, tstcc_backbone):
    if tstcc_backbone is None:
        return cand

    src_stages = [
        tstcc_backbone.conv_block1,
        tstcc_backbone.conv_block2,
        tstcc_backbone.conv_block3,
    ]

    for dst_stage, src_stage in zip(cand.encoder.blocks, src_stages):
        src_conv = next((m for m in src_stage.modules() if isinstance(m, nn.Conv1d)), None)
        src_bn = next((m for m in src_stage.modules() if isinstance(m, nn.BatchNorm1d)), None)
        dst_conv = next((m for m in dst_stage.modules() if isinstance(m, nn.Conv1d)), None)
        dst_bn = next((m for m in dst_stage.modules() if isinstance(m, nn.BatchNorm1d)), None)

        if src_conv is not None and dst_conv is not None:
            _copy_conv1d_partial(dst_conv, src_conv)
        if src_bn is not None and dst_bn is not None:
            _copy_bn1d_partial(dst_bn, src_bn)

    return cand


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
    warmup_params = [
        p for n, p in cand.named_parameters()
        if p.requires_grad and "arch_params" not in n
    ]
    opt = torch.optim.Adam(warmup_params, lr=lr)
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


def normalize_optional_limit(max_fit):
    if max_fit is None:
        return None
    max_fit = int(max_fit)
    return max_fit if max_fit > 0 else None


def build_oneclass_config(args, *, epochs=None, warmup_epochs=None, max_fit=None):
    return OneClassConfig(
        method=args.oneclass_method,
        epochs=int(args.oneclass_epochs if epochs is None else epochs),
        lr=float(args.oneclass_lr),
        batch_size=int(args.oneclass_batch_size),
        max_fit=normalize_optional_limit(args.oneclass_max_fit if max_fit is None else max_fit),
        knn_k=int(args.knn_k),
        ocsvm_nu=float(args.ocsvm_nu),
        ocsvm_kernel=str(args.ocsvm_kernel),
        ocsvm_gamma=str(args.ocsvm_gamma),
        ocsvm_degree=int(args.ocsvm_degree),
        ocsvm_coef0=float(args.ocsvm_coef0),
        svdd_hidden_dim=int(args.svdd_hidden_dim),
        svdd_rep_dim=int(args.svdd_rep_dim),
        svdd_nu=float(args.svdd_nu),
        svdd_warmup_epochs=int(args.svdd_warmup_epochs if warmup_epochs is None else warmup_epochs),
        ae_hidden_dim=int(args.ae_hidden_dim),
        ae_latent_dim=int(args.ae_latent_dim),
        maha_hidden_dim=int(args.maha_hidden_dim),
        maha_rep_dim=int(args.maha_rep_dim),
        maha_shrinkage=float(args.maha_shrinkage),
        gmm_hidden_dim=int(args.gmm_hidden_dim),
        gmm_rep_dim=int(args.gmm_rep_dim),
        gmm_components=int(args.gmm_components),
        gmm_covariance_type=str(args.gmm_covariance_type),
        gmm_reg_covar=float(args.gmm_reg_covar),
        gmm_warmup_epochs=int(args.gmm_warmup_epochs),
        proto_hidden_dim=int(args.proto_hidden_dim),
        proto_rep_dim=int(args.proto_rep_dim),
        proto_count=int(args.proto_count),
        proto_separation_weight=float(args.proto_separation_weight),
        proto_separation_margin=float(args.proto_separation_margin),
    )


def subsample_array(X_np, max_items=None, seed=42):
    max_items = normalize_optional_limit(max_items)
    if max_items is None or len(X_np) <= max_items:
        return X_np
    idx = np.random.RandomState(seed).choice(len(X_np), size=max_items, replace=False)
    return X_np[idx]


def fit_oneclass_on_features(Zs_np, device, oneclass_cfg: OneClassConfig, seed=42):
    backend = build_oneclass_backend(in_dim=Zs_np.shape[1], config=oneclass_cfg)
    backend.fit(Zs_np, device=device, seed=seed)
    backend.eval()
    return backend


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
def score_candidate_oneclass_stream(cand, backend, X_np, device, batch_size=256):
    cand.eval()
    backend.eval()
    scores = []

    dl = DataLoader(ArrayDataset(X_np), batch_size=batch_size, shuffle=False)
    for xb in dl:
        xb = xb.to(device)
        f = cand.forward_features(xb)  # [B, D]
        sc = backend.score_tensor(f)
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

    if np.unique(y_true).size < 2:
        return float("nan")

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
    Xs, Ys, X_target_pool,
    X_source_holdout,
    X_eval, Y_eval,
    args,
    device,
    in_ch,
    N_ITERS,
    tstcc_backbone,
    seed,
    out_dir="outputs",
):
    """
    FINAL-ONLY for combined mode:
      warmup on source -> freeze -> one-class model on forward_features (fit on source normal)
      -> weights on target_pool_unlabeled -> unlabeled bilevel final
      -> fit one-class model on adapted source features -> eval on X_eval/Y_eval (test if provided else val)
    """
    weighting_cfg = build_oneclass_config(
        args,
        epochs=args.oneclass_epochs,
        warmup_epochs=args.svdd_warmup_epochs,
    )
    final_cfg = build_oneclass_config(
        args,
        epochs=args.oneclass_final_epochs,
        warmup_epochs=args.svdd_final_warmup_epochs,
    )
    tau = 1.0
    w_min = 0.05

    model = CandidateModel(in_ch, arch_cfg, num_classes=2).to(device)
    initialize_candidate_from_tstcc(model, tstcc_backbone)

    ds_source = ArrayDataset(Xs, Ys)

    warmup_candidate_on_source(
        model,
        ds_source,
        device=device,
        steps=args.combined_final_candidate_warmup_steps,
        bs=args.batch_size,
        lr=1e-3,
    )

    _set_requires_grad(model, False)
    Xs_fit = subsample_array(Xs, weighting_cfg.max_fit, seed=seed)
    Fs = extract_candidate_features(model, Xs_fit, device=device, batch_size=256)
    weighting_backend = fit_oneclass_on_features(Fs, device=device, oneclass_cfg=weighting_cfg, seed=seed)

    raw = score_candidate_oneclass_stream(
        model,
        weighting_backend,
        X_target_pool,
        device=device,
        batch_size=256,
    )
    w_ent = robust_sigmoid_weights(raw, tau=tau, w_min=w_min)

    print(
        f"[FINAL-ONLY][{arch_name}] weights: mean={w_ent.mean():.4f} "
        f"min={w_ent.min():.4f} max={w_ent.max():.4f} | "
        f"tau={tau} w_min={w_min} method={weighting_cfg.method} score={weighting_backend.score_name}"
    )

    _set_requires_grad(model, True)
    model.train()

    ds_target = ArrayDataset(X_target_pool, None, w=w_ent)
    upper_src_loader, upper_tgt_loader = build_upper_unlabeled_loaders(
        X_source_holdout, X_target_pool, w_ent, bs=args.batch_size
    )
    alpha_final = 0.3 + 0.2 * (N_ITERS - 1)

    train_log = train_bilevel(
        model, ds_source, ds_target, None,
        device=device,
        steps=args.combined_final_steps,
        bs=args.batch_size,
        alpha=alpha_final,
        gamma=1.0,
        lr_inner=1e-3,
        lr_arch=1e-3,
        use_cosine_decay=True,
        early_stop=True,
        patience=args.combined_final_patience,
        ckpt_path=os.path.join(out_dir, "checkpoints", f"{arch_name}_final_best.pt"),
        upper_source_loader=upper_src_loader,
        upper_target_loader=upper_tgt_loader,
        upper_beta_gap=args.combined_upper_gap,
    )

    metrics_uad = None
    final_backend_info = None
    if X_eval is not None and Y_eval is not None:
        scores, train_scores, final_backend_info = fit_final_oneclass_and_score(
            model,
            X_train_norm=Xs,
            X_eval=X_eval,
            device=device,
            oneclass_cfg=final_cfg,
            seed=seed,
        )
        ap, auroc = compute_ap_auroc(Y_eval, scores)

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
        "oneclass": {
            "method": args.oneclass_method,
            "weighting_backend": weighting_backend.summary(),
            "final_backend": final_backend_info,
        },
        "weighting": {
            "option": "option2_oneclass_on_forward_features",
            "method": args.oneclass_method,
            "score_name": weighting_backend.score_name,
            "tau": float(tau),
            "w_min": float(w_min),
            "config": weighting_cfg.to_dict(),
        }
    }


# ---------------- UAD-SOURCE: one-class objective search ----------------
def oneclass_objective_on_source_normal(
    cand,
    X_train_norm,
    X_val_norm,
    device,
    oneclass_cfg: OneClassConfig,
    seed=42,
):
    """
    Objective to MINIMIZE:
      - fit one-class model on train normal features (candidate forward_features)
      - compute mean(score) on val normal
    Higher score means more anomalous, so lower mean score means tighter normal compactness.
    """
    _set_requires_grad(cand, False)
    cand.eval()

    X_fit = subsample_array(X_train_norm, oneclass_cfg.max_fit, seed=seed)
    F_fit = extract_candidate_features(cand, X_fit, device=device, batch_size=256)
    backend = fit_oneclass_on_features(F_fit, device=device, oneclass_cfg=oneclass_cfg, seed=seed)

    F_val = extract_candidate_features(cand, X_val_norm, device=device, batch_size=256)
    with torch.no_grad():
        score_vals = backend.score_tensor(
            torch.tensor(F_val, dtype=torch.float32, device=device)
        ).detach().cpu().numpy()

    obj = float(score_vals.mean())
    info = {
        "mean_score": obj,
        "score_name": backend.score_name,
        "oneclass": backend.summary(),
    }
    if backend.method_name == "deepsvdd":
        info["mean_dist2"] = obj
    return obj, info


def fit_final_oneclass_and_score(
    cand,
    X_train_norm,
    X_eval,
    device,
    oneclass_cfg: OneClassConfig,
    seed=42,
):
    """
    Fit one-class model on full train normal-only features, then score eval data.
    Returns: scores_eval (np.ndarray), scores_train_norm (np.ndarray)
    """
    cand.eval()
    _set_requires_grad(cand, False)

    X_fit = subsample_array(X_train_norm, oneclass_cfg.max_fit, seed=seed)
    F_fit = extract_candidate_features(cand, X_fit, device=device, batch_size=256)
    backend = fit_oneclass_on_features(F_fit, device=device, oneclass_cfg=oneclass_cfg, seed=seed)

    scores_train = score_candidate_oneclass_stream(cand, backend, X_train_norm, device=device, batch_size=256)
    scores_eval = score_candidate_oneclass_stream(cand, backend, X_eval, device=device, batch_size=256)
    return scores_eval, scores_train, backend.summary()


_OMNI_PAPER_LOW_QUANTILES = {
    "smd_group_1": 0.0001,
    "smd_group_2": 0.0025,
    "smd_group_3": 0.0050,
}

_OMNI_REPO_LEVELS = {
    "smd_group_1": 0.0050,
    "smd_group_2": 0.0075,
    "smd_group_3": 0.0001,
}

_OMNI_REFERENCE_Q = {
    "paper": 1e-4,
    "repo": 1e-3,
}


def _omni_group_from_machine(machine: str) -> str:
    if machine.startswith("machine-1-"):
        return "smd_group_1"
    if machine.startswith("machine-2-"):
        return "smd_group_2"
    if machine.startswith("machine-3-"):
        return "smd_group_3"
    return "unknown"


def resolve_omni_reference_settings(machine: str, args):
    """
    Resolve POT settings for OmniAnomaly in either paper-faithful or repo-faithful mode.

    Notes:
    - Paper appendix (KDD'19) reports q=1e-4 and SMD low quantiles:
      group1=0.0001, group2=0.0025, group3=0.005.
    - The public OmniAnomaly repo comments recommend:
      group1=0.005, group2=0.0075, group3=0.0001 with q=1e-3.
    """
    ref = getattr(args, "omni_reference", "paper")
    group = _omni_group_from_machine(machine)

    if ref == "repo":
        auto_level = _OMNI_REPO_LEVELS.get(group, 0.01)
    else:
        auto_level = _OMNI_PAPER_LOW_QUANTILES.get(group, 0.005)

    auto_q = _OMNI_REFERENCE_Q.get(ref, 1e-4)

    pot_q = float(args.omni_pot_q) if args.omni_pot_q and args.omni_pot_q > 0 else float(auto_q)
    pot_level = float(args.omni_pot_level) if args.omni_pot_level and args.omni_pot_level > 0 else float(auto_level)

    return {
        "reference": ref,
        "group": group,
        "pot_q": pot_q,
        "pot_level": pot_level,
        "pot_q_source": "cli" if args.omni_pot_q and args.omni_pot_q > 0 else ref,
        "pot_level_source": "cli" if args.omni_pot_level and args.omni_pot_level > 0 else ref,
    }


def _generic_uad_metrics_from_scores(y_eval, scores_eval, scores_train, *, pot_q=1e-3, pot_level=0.99):
    ap, auroc = compute_ap_auroc(y_eval, scores_eval)
    thr_pot = pot_threshold(scores_train, q=pot_q, level=pot_level)
    p_pot, r_pot, f1_pot = f1_at_threshold(y_eval, scores_eval, thr_pot)
    f1_b, p_b, r_b, thr_b = best_f1(y_eval, scores_eval)
    y_pred_bin = (np.asarray(scores_eval) >= float(thr_pot)).astype(int)
    ev = event_f1_and_delay(y_eval, y_pred_bin)
    return {
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


def _omni_metrics_from_scores(y_eval, scores_eval, scores_train, *, pot_q=1e-3, pot_level=0.98):
    ap, auroc = compute_ap_auroc(y_eval, scores_eval)
    normal_scores_eval = -np.asarray(scores_eval).astype(float)
    normal_scores_train = -np.asarray(scores_train).astype(float)

    bf_res = omni_bf_search(
        normal_scores_eval,
        y_eval,
        start=float(normal_scores_eval.min()),
        end=float(normal_scores_eval.max()),
        step_num=100,
        display_freq=200,
        verbose=False,
    )
    pot_res = omni_pot_eval(
        normal_scores_train,
        normal_scores_eval,
        y_eval,
        q=pot_q,
        level=pot_level,
    )

    thr_pot = float(-pot_res["pot-threshold"])
    p_pot = float(pot_res["pot-precision"])
    r_pot = float(pot_res["pot-recall"])
    f1_pot = float(pot_res["pot-f1"])
    f1_b = float(bf_res["f1"])
    p_b = float(bf_res["precision"])
    r_b = float(bf_res["recall"])
    thr_b = float(-bf_res["threshold"])
    y_pred_bin = np.asarray(pot_res["pred"]).astype(int)
    ev = event_f1_and_delay(y_eval, y_pred_bin)
    return {
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
        "latency_best": float(bf_res["latency"]),
        "latency_pot": float(pot_res["pot-latency"]),
        "event_f1": float(ev["event_f1"]),
        "event_precision": float(ev["event_precision"]),
        "event_recall": float(ev["event_recall"]),
        "delay_mean": float(ev["delay_mean"]),
        "delay_median": float(ev["delay_median"]),
    }


def _evaluate_omni_arch_on_raw_source(
    arch,
    *,
    in_ch,
    x_train_inner,
    x_val_inner,
    x_train_full,
    x_test,
    y_test_aligned,
    device,
    args,
    pot_q,
    pot_level,
):
    model = OmniAnomalyModel(in_ch, arch).to(device)
    search_log = train_omni_source(
        model,
        x_train_inner,
        x_val_inner,
        device=device,
        arch=arch,
        epochs=args.omni_epochs,
        patience=args.omni_patience,
    )
    val_stats = validate_omni_on_series(
        model,
        x_val_inner,
        device=device,
        batch_size=arch.batch_size,
        window_length=arch.window_length,
        stride=arch.stride,
        n_z=arch.test_n_z,
    )

    final_model = OmniAnomalyModel(in_ch, arch).to(device)
    final_model.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    final_log = train_omni_source(
        final_model,
        x_train_full,
        None,
        device=device,
        arch=arch,
        epochs=args.omni_final_epochs,
        patience=max(2, args.omni_patience),
    )

    scores_train = score_omni_series(
        final_model,
        x_train_full,
        device=device,
        batch_size=arch.batch_size,
        window_length=arch.window_length,
        stride=arch.stride,
        n_z=arch.test_n_z,
    )
    scores_test = score_omni_series(
        final_model,
        x_test,
        device=device,
        batch_size=arch.batch_size,
        window_length=arch.window_length,
        stride=arch.stride,
        n_z=arch.test_n_z,
    )

    return {
        "arch": arch,
        "val_stats": val_stats,
        "search_log": search_log,
        "final_log": final_log,
        "scores_train": scores_train,
        "scores_test": scores_test,
        "metrics_uad": _omni_metrics_from_scores(
            y_test_aligned,
            scores_test,
            scores_train,
            pot_q=pot_q,
            pot_level=pot_level,
        ),
    }


def _evaluate_usad_arch_on_raw_source(
    arch,
    *,
    train_windows_inner,
    val_windows_inner,
    train_windows_full,
    test_windows,
    y_test_window_labels,
    device,
    args,
    pot_q,
    pot_level,
):
    w_size = int(train_windows_full.shape[1])

    model = UsadModel(w_size, arch).to(device)
    search_log = train_usad_source(
        model,
        train_windows_inner,
        val_windows_inner,
        device=device,
        arch=arch,
        epochs=args.usad_epochs,
        patience=args.usad_patience,
        shuffle=False,
        use_early_stopping=False,
        restore_best_state=False,
    )
    val_stats = validate_usad_on_windows(
        model,
        val_windows_inner,
        device=device,
        batch_size=arch.batch_size,
        epoch_idx=args.usad_epochs,
    )

    final_model = UsadModel(w_size, arch).to(device)
    final_model.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    final_log = train_usad_source(
        final_model,
        train_windows_full,
        val_windows_inner,
        device=device,
        arch=arch,
        epochs=args.usad_final_epochs,
        patience=max(1, args.usad_final_patience if args.usad_final_patience > 0 else args.usad_patience),
        shuffle=False,
        use_early_stopping=bool(args.usad_final_early_stopping),
        restore_best_state=bool(args.usad_final_early_stopping),
    )

    scores_train = score_usad_windows(
        final_model,
        train_windows_full,
        device=device,
        batch_size=arch.batch_size,
    )
    scores_test = score_usad_windows(
        final_model,
        test_windows,
        device=device,
        batch_size=arch.batch_size,
    )

    return {
        "arch": arch,
        "val_stats": val_stats,
        "search_log": search_log,
        "final_log": final_log,
        "scores_train": scores_train,
        "scores_test": scores_test,
        "metrics_uad": _generic_uad_metrics_from_scores(
            y_test_window_labels,
            scores_test,
            scores_train,
            pot_q=pot_q,
            pot_level=pot_level,
        ),
    }


def _evaluate_tranad_arch_on_raw_source(
    arch,
    *,
    in_ch,
    train_windows_inner,
    val_windows_inner,
    train_windows_full,
    test_windows,
    y_test_labels,
    device,
    args,
    pot_q,
    pot_level,
):
    model = TranADModel(in_ch, arch).to(device).double()
    search_log = train_tranad_source(
        model,
        train_windows_inner,
        val_windows_inner,
        device=device,
        arch=arch,
        epochs=args.tranad_epochs,
        patience=args.tranad_patience,
        shuffle=False,
        use_early_stopping=False,
        restore_best_state=False,
    )
    val_stats = validate_tranad_on_windows(
        model,
        val_windows_inner,
        device=device,
        batch_size=arch.batch_size,
        epoch_idx=args.tranad_epochs,
    )

    final_model = TranADModel(in_ch, arch).to(device).double()
    final_model.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    final_log = train_tranad_source(
        final_model,
        train_windows_full,
        None,
        device=device,
        arch=arch,
        epochs=args.tranad_final_epochs,
        patience=max(1, args.tranad_patience),
        shuffle=False,
        use_early_stopping=False,
        restore_best_state=False,
    )

    scores_train = score_tranad_windows(
        final_model,
        train_windows_full,
        device=device,
        batch_size=arch.batch_size,
    )
    scores_test = score_tranad_windows(
        final_model,
        test_windows,
        device=device,
        batch_size=arch.batch_size,
    )

    return {
        "arch": arch,
        "val_stats": val_stats,
        "search_log": search_log,
        "final_log": final_log,
        "scores_train": scores_train,
        "scores_test": scores_test,
        "metrics_uad": _generic_uad_metrics_from_scores(
            y_test_labels,
            scores_test,
            scores_train,
            pot_q=pot_q,
            pot_level=pot_level,
        ),
    }


def run_tranad_uad_source_family_raw(*, raw_smd_root, machine, device, args):
    raw_smd_root = resolve_raw_smd_root(raw_smd_root)
    x_train, x_test, y_test = load_raw_tranad_smd_machine(raw_smd_root, machine)

    train_start = max(0, int(getattr(args, "tranad_train_start", 0)))
    test_start = max(0, int(getattr(args, "tranad_test_start", 0)))
    if args.tranad_train_limit and args.tranad_train_limit > 0:
        x_train = x_train[train_start:train_start + args.tranad_train_limit]
    elif train_start > 0:
        x_train = x_train[train_start:]
    if args.tranad_test_limit and args.tranad_test_limit > 0:
        x_test = x_test[test_start:test_start + args.tranad_test_limit]
        y_test = y_test[test_start:test_start + args.tranad_test_limit]
    elif test_start > 0:
        x_test = x_test[test_start:]
        y_test = y_test[test_start:]

    in_ch = x_train.shape[-1]
    fixed_arch = get_fixed_paper_tranad_arch(window_length=args.tranad_window_length)
    fixed_arch = replace(
        fixed_arch,
        ff_dim=args.tranad_ff_dim if args.tranad_ff_dim > 0 else fixed_arch.ff_dim,
        dropout=args.tranad_dropout,
        encoder_layers=args.tranad_encoder_layers if args.tranad_encoder_layers > 0 else fixed_arch.encoder_layers,
        decoder_layers=args.tranad_decoder_layers if args.tranad_decoder_layers > 0 else fixed_arch.decoder_layers,
        batch_size=args.tranad_batch_size,
        max_epoch=args.tranad_epochs,
        valid_ratio=args.tranad_valid_ratio,
        lr=args.tranad_lr,
    )

    train_windows_full = build_tranad_windows(x_train, fixed_arch.window_length)
    x_train_inner, x_val_inner = contiguous_train_valid_split(
        train_windows_full,
        valid_ratio=fixed_arch.valid_ratio,
    )
    test_windows = build_tranad_windows(x_test, fixed_arch.window_length)

    print("[TRANAD RAW INPUT]")
    print("  machine     :", machine)
    print("  train raw   :", x_train.shape)
    print("  test raw    :", x_test.shape)
    print("  labels test :", y_test.shape, "positives:", int(np.sum(y_test)))
    print("  train win   :", x_train_inner.shape, "| val win:", x_val_inner.shape)
    print("  test win    :", test_windows.shape)
    print("  window      :", fixed_arch.window_length)
    print("  ff/dropout  :", fixed_arch.ff_dim, fixed_arch.dropout)
    print("  enc/dec     :", fixed_arch.encoder_layers, fixed_arch.decoder_layers)
    print("  POT q/level :", args.tranad_pot_q, args.tranad_pot_level)

    print("\n[FIXED BASELINE] TranAD upstream-faithful fixed architecture...")
    fixed_eval = _evaluate_tranad_arch_on_raw_source(
        fixed_arch,
        in_ch=in_ch,
        train_windows_inner=x_train_inner,
        val_windows_inner=x_val_inner,
        train_windows_full=train_windows_full,
        test_windows=test_windows,
        y_test_labels=y_test,
        device=device,
        args=args,
        pot_q=args.tranad_pot_q,
        pot_level=args.tranad_pot_level,
    )

    if getattr(args, "tranad_fixed_only", False):
        return {
            "mode": "uad_source",
            "family": "tranad",
            "protocol": "raw_smd_machine_by_machine",
            "machine": machine,
            "raw_smd_root": str(raw_smd_root),
            "best_arch": str(fixed_arch),
            "search_history": [],
            "tranad_family_notes": {
                "target": "upstream_style_source_only",
                "dataset": "SMD",
                "scoring": "phase2_last_step_forecasting_mse",
                "training": "epoch_weighted_phase1_phase2_reconstruction_loss",
                "preprocess_mode": "none",
                "frozen_core_components": [
                    "raw_smd_no_normalization",
                    "front_padded_window_protocol",
                    "one_transformer_encoder",
                    "two_transformer_decoders",
                    "self_conditioning_phase2",
                    "custom_repo_transformer_blocks_without_layernorm",
                    "paper_style_epoch_weighted_phase_loss",
                    "last_step_forecasting_mse_score",
                ],
                "searched_components": [
                    "ff_dim",
                    "dropout",
                    "encoder_layers",
                    "decoder_layers",
                ],
                "fixed_only": True,
            },
            "fixed_baseline": {
                "arch": str(fixed_arch),
                "metrics_uad": fixed_eval["metrics_uad"],
                "val_stats": fixed_eval["val_stats"],
                "search_train_curve": fixed_eval["search_log"],
                "final_train_curve": fixed_eval["final_log"],
            },
            "searched_partial_nas": None,
            "metrics_uad": fixed_eval["metrics_uad"],
        }

    n_iters = max(1, int(args.tranad_search_iters))
    history = []
    best_obj = float("inf")
    best_arch = None
    best_eval = None

    for iter_id in range(n_iters):
        print(f"\n[ITER {iter_id + 1}/{n_iters}] TranAD partial NAS on raw SMD...")
        for i in range(args.search_candidates):
            arch_c = sample_tranad_arch(window_length=args.tranad_window_length)
            arch_c = replace(
                arch_c,
                batch_size=args.tranad_batch_size,
                max_epoch=args.tranad_epochs,
                valid_ratio=args.tranad_valid_ratio,
                lr=args.tranad_lr,
            )
            eval_out = _evaluate_tranad_arch_on_raw_source(
                arch_c,
                in_ch=in_ch,
                train_windows_inner=x_train_inner,
                val_windows_inner=x_val_inner,
                train_windows_full=train_windows_full,
                test_windows=test_windows,
                y_test_labels=y_test,
                device=device,
                args=args,
                pot_q=args.tranad_pot_q,
                pot_level=args.tranad_pot_level,
            )
            obj = float(eval_out["val_stats"]["val_score"])
            history.append(
                {
                    "iter": iter_id + 1,
                    "arch": str(arch_c),
                    "objective": obj,
                    "val_phase1_mse": float(eval_out["val_stats"]["val_phase1_mse"]),
                    "val_phase2_mse": float(eval_out["val_stats"]["val_phase2_mse"]),
                    "val_loss": float(eval_out["val_stats"]["val_loss"]),
                    "val_score": float(eval_out["val_stats"]["val_score"]),
                }
            )
            print(f"  Candidate {i + 1}/{args.search_candidates}: obj(val_score)={obj:.6f}")

            if obj < best_obj:
                best_obj = obj
                best_arch = arch_c
                best_eval = eval_out

    if best_arch is None or best_eval is None:
        raise RuntimeError("tranad/uad_source/raw: best_arch is None after search.")

    print(f"\n[TRANAD SEARCH DONE] Best partial-NAS arch = {best_arch} | best_obj={best_obj:.6f}")

    return {
        "mode": "uad_source",
        "family": "tranad",
        "protocol": "raw_smd_machine_by_machine",
        "machine": machine,
        "raw_smd_root": str(raw_smd_root),
        "best_arch": str(best_arch),
        "search_history": history,
        "tranad_family_notes": {
            "target": "upstream_style_source_only",
            "dataset": "SMD",
            "scoring": "phase2_last_step_forecasting_mse",
            "training": "epoch_weighted_phase1_phase2_reconstruction_loss",
            "preprocess_mode": "none",
            "frozen_core_components": [
                "raw_smd_no_normalization",
                "front_padded_window_protocol",
                "one_transformer_encoder",
                "two_transformer_decoders",
                "self_conditioning_phase2",
                "custom_repo_transformer_blocks_without_layernorm",
                "paper_style_epoch_weighted_phase_loss",
                "last_step_forecasting_mse_score",
            ],
            "searched_components": [
                "ff_dim",
                "dropout",
                "encoder_layers",
                "decoder_layers",
            ],
        },
        "fixed_baseline": {
            "arch": str(fixed_arch),
            "metrics_uad": fixed_eval["metrics_uad"],
            "val_stats": fixed_eval["val_stats"],
            "search_train_curve": fixed_eval["search_log"],
            "final_train_curve": fixed_eval["final_log"],
        },
        "searched_partial_nas": {
            "arch": str(best_arch),
            "metrics_uad": best_eval["metrics_uad"],
            "val_stats": best_eval["val_stats"],
            "search_train_curve": best_eval["search_log"],
            "final_train_curve": best_eval["final_log"],
        },
        "metrics_uad": best_eval["metrics_uad"],
    }


def run_usad_uad_source_family_raw(*, swat_train_csv, swat_test_csv, device, args):
    swat_data = RawSWaTDataset.from_csvs(
        swat_train_csv,
        swat_test_csv,
        preprocess_mode=args.usad_preprocess,
        downsample=args.usad_downsample,
    )
    if args.usad_train_limit and args.usad_train_limit > 0:
        swat_data.x_train = swat_data.x_train[:args.usad_train_limit]
    if args.usad_test_limit and args.usad_test_limit > 0:
        swat_data.x_test = swat_data.x_test[:args.usad_test_limit]
        swat_data.y_test = swat_data.y_test[:args.usad_test_limit]

    fixed_arch = get_fixed_paper_usad_arch(
        window_length=args.usad_window_length,
        downsample=args.usad_downsample,
    )
    fixed_arch = replace(
        fixed_arch,
        latent_size=args.usad_latent_size if args.usad_latent_size > 0 else fixed_arch.latent_size,
        batch_size=args.usad_batch_size,
        max_epoch=args.usad_epochs,
        valid_ratio=args.usad_valid_ratio,
        lr=args.usad_lr,
        stride=args.usad_stride,
        score_alpha=args.usad_score_alpha,
        score_beta=args.usad_score_beta,
    )

    train_windows_full = build_upstream_usad_flat_windows(
        swat_data.x_train,
        fixed_arch.window_length,
        fixed_arch.stride,
    )
    x_train_inner, x_val_inner = contiguous_train_valid_split(
        train_windows_full,
        valid_ratio=fixed_arch.valid_ratio,
    )
    test_windows = build_upstream_usad_flat_windows(
        swat_data.x_test,
        fixed_arch.window_length,
        fixed_arch.stride,
    )
    y_test_aligned = build_upstream_usad_window_labels(
        swat_data.y_test,
        window_length=fixed_arch.window_length,
        stride=fixed_arch.stride,
    )

    print("[USAD RAW INPUT]")
    print("  train csv   :", swat_train_csv)
    print("  test csv    :", swat_test_csv)
    print("  train raw   :", swat_data.x_train.shape)
    print("  test raw    :", swat_data.x_test.shape)
    print("  train win   :", x_train_inner.shape, "| val win:", x_val_inner.shape)
    print("  test win    :", test_windows.shape)
    print("  labels test :", swat_data.y_test.shape, "window-any:", y_test_aligned.shape)
    print("  window      :", fixed_arch.window_length, "stride:", fixed_arch.stride)
    print("  downsample  :", fixed_arch.downsample)
    print("  preprocess  :", args.usad_preprocess)
    print("  score a/b   :", fixed_arch.score_alpha, fixed_arch.score_beta)
    print("  POT q/level :", args.usad_pot_q, args.usad_pot_level)

    print("\n[FIXED BASELINE] USAD paper-style fixed architecture...")
    fixed_eval = _evaluate_usad_arch_on_raw_source(
        fixed_arch,
        train_windows_inner=x_train_inner,
        val_windows_inner=x_val_inner,
        train_windows_full=train_windows_full,
        test_windows=test_windows,
        y_test_window_labels=y_test_aligned,
        device=device,
        args=args,
        pot_q=args.usad_pot_q,
        pot_level=args.usad_pot_level,
    )

    if getattr(args, "usad_fixed_only", False):
        return {
            "mode": "uad_source",
            "family": "usad",
            "protocol": "raw_swat_normal_attack",
            "swat_train_csv": str(swat_train_csv),
            "swat_test_csv": str(swat_test_csv),
            "best_arch": str(fixed_arch),
            "search_history": [],
            "usad_family_notes": {
                "target": "paper_style_source_only",
                "dataset": "SWaT",
                "scoring": "alpha_mse(x,w1)+beta_mse(x,w3)",
                "training": "two_optimizer_adversarial_autoencoder",
                "preprocess_mode": args.usad_preprocess,
                "downsample": args.usad_downsample,
                "frozen_core_components": [
                    "one_mlp_encoder",
                    "two_mlp_decoders",
                    "three_linear_layers_per_module",
                    "relu_hidden_activation",
                    "sigmoid_decoder_output",
                    "paper_style_usad_loss1_loss2",
                    "paper_style_usad_anomaly_score",
                ],
                "searched_components": [
                    "latent_size",
                    "hidden_scale",
                ],
                "fixed_only": True,
            },
            "fixed_baseline": {
                "arch": str(fixed_arch),
                "metrics_uad": fixed_eval["metrics_uad"],
                "val_stats": fixed_eval["val_stats"],
                "search_train_curve": fixed_eval["search_log"],
                "final_train_curve": fixed_eval["final_log"],
            },
            "searched_partial_nas": None,
            "metrics_uad": fixed_eval["metrics_uad"],
        }

    N_ITERS = max(1, int(args.usad_search_iters))
    history = []
    best_obj = float("inf")
    best_arch = None
    best_eval = None

    for iter_id in range(N_ITERS):
        print(f"\n[ITER {iter_id+1}/{N_ITERS}] USAD UAD_SOURCE partial NAS on raw SWaT...")
        for i in range(args.search_candidates):
            arch_c = sample_usad_arch(
                window_length=args.usad_window_length,
                downsample=args.usad_downsample,
            )
            arch_c = replace(
                arch_c,
                batch_size=args.usad_batch_size,
                max_epoch=args.usad_epochs,
                valid_ratio=args.usad_valid_ratio,
                lr=args.usad_lr,
                stride=args.usad_stride,
                score_alpha=args.usad_score_alpha,
                score_beta=args.usad_score_beta,
            )
            eval_out = _evaluate_usad_arch_on_raw_source(
                arch_c,
                train_windows_inner=x_train_inner,
                val_windows_inner=x_val_inner,
                train_windows_full=train_windows_full,
                test_windows=test_windows,
                y_test_window_labels=y_test_aligned,
                device=device,
                args=args,
                pot_q=args.usad_pot_q,
                pot_level=args.usad_pot_level,
            )
            obj = float(eval_out["val_stats"]["val_score"])
            history.append(
                {
                    "iter": iter_id + 1,
                    "arch": str(arch_c),
                    "objective": obj,
                    "val_loss1": float(eval_out["val_stats"]["val_loss1"]),
                    "val_loss2": float(eval_out["val_stats"]["val_loss2"]),
                    "val_score": float(eval_out["val_stats"]["val_score"]),
                }
            )
            print(f"  Candidate {i+1}/{args.search_candidates}: obj(val_score)={obj:.6f}")

            if obj < best_obj:
                best_obj = obj
                best_arch = arch_c
                best_eval = eval_out

    if best_arch is None or best_eval is None:
        raise RuntimeError("usad/uad_source/raw: best_arch is None after search.")

    print(f"\n[USAD SEARCH DONE] Best partial-NAS arch = {best_arch} | best_obj={best_obj:.6f}")

    return {
        "mode": "uad_source",
        "family": "usad",
        "protocol": "raw_swat_normal_attack",
        "swat_train_csv": str(swat_train_csv),
        "swat_test_csv": str(swat_test_csv),
        "best_arch": str(best_arch),
        "search_history": history,
        "usad_family_notes": {
            "target": "paper_style_source_only",
            "dataset": "SWaT",
            "scoring": "alpha_mse(x,w1)+beta_mse(x,w3)",
            "training": "two_optimizer_adversarial_autoencoder",
            "preprocess_mode": args.usad_preprocess,
            "downsample": args.usad_downsample,
            "frozen_core_components": [
                "one_mlp_encoder",
                "two_mlp_decoders",
                "three_linear_layers_per_module",
                "relu_hidden_activation",
                "sigmoid_decoder_output",
                "paper_style_usad_loss1_loss2",
                "paper_style_usad_anomaly_score",
            ],
            "searched_components": [
                "latent_size",
                "hidden_scale",
            ],
        },
        "fixed_baseline": {
            "arch": str(fixed_arch),
            "metrics_uad": fixed_eval["metrics_uad"],
            "val_stats": fixed_eval["val_stats"],
            "search_train_curve": fixed_eval["search_log"],
            "final_train_curve": fixed_eval["final_log"],
        },
        "searched_partial_nas": {
            "arch": str(best_arch),
            "metrics_uad": best_eval["metrics_uad"],
            "val_stats": best_eval["val_stats"],
            "search_train_curve": best_eval["search_log"],
            "final_train_curve": best_eval["final_log"],
        },
        "metrics_uad": best_eval["metrics_uad"],
    }


def run_omni_uad_source_family_raw(*, raw_smd_root, machine, device, args):
    raw_smd_root = resolve_raw_smd_root(raw_smd_root)
    machine_data = RawSMDMachine.from_root(
        raw_smd_root,
        machine,
        preprocess_mode=args.omni_preprocess,
    )
    if args.omni_train_limit and args.omni_train_limit > 0:
        machine_data.x_train = machine_data.x_train[:args.omni_train_limit]
    if args.omni_test_limit and args.omni_test_limit > 0:
        machine_data.x_test = machine_data.x_test[:args.omni_test_limit]
        machine_data.y_test = machine_data.y_test[:args.omni_test_limit]
    in_ch = machine_data.x_train.shape[-1]

    fixed_arch = get_fixed_paper_omni_arch(window_length=args.omni_window_length)
    fixed_arch = replace(
        fixed_arch,
        batch_size=args.omni_batch_size,
        max_epoch=args.omni_epochs,
        lr=args.omni_lr,
        valid_ratio=args.omni_valid_ratio,
        stride=args.omni_stride,
        test_n_z=args.omni_test_n_z,
    )

    x_train_inner, x_val_inner = contiguous_train_valid_split(
        machine_data.x_train,
        valid_ratio=fixed_arch.valid_ratio,
    )
    y_test_aligned = aligned_last_point_labels(
        machine_data.y_test,
        window_length=fixed_arch.window_length,
        stride=fixed_arch.stride,
    )
    omni_ref = resolve_omni_reference_settings(machine, args)

    print("[OMNI RAW INPUT]")
    print("  machine     :", machine)
    print("  train raw   :", machine_data.x_train.shape)
    print("  test raw    :", machine_data.x_test.shape)
    print("  labels test :", machine_data.y_test.shape, "aligned:", y_test_aligned.shape)
    print("  window      :", fixed_arch.window_length, "stride:", fixed_arch.stride)
    print("  omni ref    :", omni_ref["reference"], "| group:", omni_ref["group"])
    print("  POT q/level :", omni_ref["pot_q"], omni_ref["pot_level"])
    print("  preprocess  :", args.omni_preprocess)

    print("\n[FIXED BASELINE] OmniAnomaly paper-faithful fixed architecture...")
    fixed_eval = _evaluate_omni_arch_on_raw_source(
        fixed_arch,
        in_ch=in_ch,
        x_train_inner=x_train_inner,
        x_val_inner=x_val_inner,
        x_train_full=machine_data.x_train,
        x_test=machine_data.x_test,
        y_test_aligned=y_test_aligned,
        device=device,
        args=args,
        pot_q=omni_ref["pot_q"],
        pot_level=omni_ref["pot_level"],
    )

    if getattr(args, "omni_fixed_only", False):
        return {
            "mode": "uad_source",
            "family": "omni_anomaly",
            "protocol": "raw_smd_machine_by_machine",
            "machine": machine,
            "raw_smd_root": str(raw_smd_root),
            "best_arch": str(fixed_arch),
            "search_history": [],
            "omni_family_notes": {
                "target": "paper_faithful_source_only",
                "scoring": "negative_last_point_reconstruction_log_probability",
                "posterior_flow": "planar_nf",
                "ts_tcc": "disabled_for_omni_family",
                "deepsvdd": "disabled_for_omni_family",
                "omni_reference": omni_ref["reference"],
                "omni_group": omni_ref["group"],
                "pot_q": omni_ref["pot_q"],
                "pot_level": omni_ref["pot_level"],
                "pot_q_source": omni_ref["pot_q_source"],
                "pot_level_source": omni_ref["pot_level_source"],
                "preprocess_mode": args.omni_preprocess,
                "fixed_only": True,
            },
            "fixed_baseline": {
                "arch": str(fixed_arch),
                "metrics_uad": fixed_eval["metrics_uad"],
                "val_stats": fixed_eval["val_stats"],
                "search_train_curve": fixed_eval["search_log"],
                "final_train_curve": fixed_eval["final_log"],
            },
            "searched_partial_nas": None,
            "metrics_uad": fixed_eval["metrics_uad"],
        }

    N_ITERS = max(1, int(args.omni_search_iters))
    history = []
    best_obj = float("inf")
    best_arch = None
    best_eval = None

    for iter_id in range(N_ITERS):
        print(f"\n[ITER {iter_id+1}/{N_ITERS}] OMNI UAD_SOURCE partial NAS on raw SMD...")
        for i in range(args.search_candidates):
            arch_c = sample_omni_arch(window_length=args.omni_window_length)
            arch_c = replace(
                arch_c,
                batch_size=args.omni_batch_size,
                max_epoch=args.omni_epochs,
                lr=args.omni_lr,
                valid_ratio=args.omni_valid_ratio,
                stride=args.omni_stride,
                test_n_z=args.omni_test_n_z,
            )
            eval_out = _evaluate_omni_arch_on_raw_source(
                arch_c,
                in_ch=in_ch,
                x_train_inner=x_train_inner,
                x_val_inner=x_val_inner,
                x_train_full=machine_data.x_train,
                x_test=machine_data.x_test,
                y_test_aligned=y_test_aligned,
                device=device,
                args=args,
                pot_q=omni_ref["pot_q"],
                pot_level=omni_ref["pot_level"],
            )
            obj = float(eval_out["val_stats"]["val_score_last"])
            history.append({
                "iter": iter_id + 1,
                "arch": str(arch_c),
                "objective": obj,
                "val_loss": float(eval_out["val_stats"]["val_loss"]),
                "val_score_last": float(eval_out["val_stats"]["val_score_last"]),
            })
            print(f"  Candidate {i+1}/{args.search_candidates}: obj(val_score_last)={obj:.6f}")

            if obj < best_obj:
                best_obj = obj
                best_arch = arch_c
                best_eval = eval_out

    if best_arch is None or best_eval is None:
        raise RuntimeError("omni_anomaly/uad_source/raw: best_arch is None after search.")

    print(f"\n[OMNI SEARCH DONE] Best partial-NAS arch = {best_arch} | best_obj={best_obj:.6f}")

    return {
        "mode": "uad_source",
        "family": "omni_anomaly",
        "protocol": "raw_smd_machine_by_machine",
        "machine": machine,
        "raw_smd_root": str(raw_smd_root),
        "best_arch": str(best_arch),
        "search_history": history,
        "omni_family_notes": {
            "target": "paper_faithful_source_only",
            "scoring": "negative_last_point_reconstruction_log_probability",
            "posterior_flow": "planar_nf",
            "ts_tcc": "disabled_for_omni_family",
            "deepsvdd": "disabled_for_omni_family",
            "omni_reference": omni_ref["reference"],
            "omni_group": omni_ref["group"],
            "pot_q": omni_ref["pot_q"],
            "pot_level": omni_ref["pot_level"],
            "pot_q_source": omni_ref["pot_q_source"],
            "pot_level_source": omni_ref["pot_level_source"],
            "preprocess_mode": args.omni_preprocess,
        },
        "fixed_baseline": {
            "arch": str(fixed_arch),
            "metrics_uad": fixed_eval["metrics_uad"],
            "val_stats": fixed_eval["val_stats"],
            "search_train_curve": fixed_eval["search_log"],
            "final_train_curve": fixed_eval["final_log"],
        },
        "searched_partial_nas": {
            "arch": str(best_arch),
            "metrics_uad": best_eval["metrics_uad"],
            "val_stats": best_eval["val_stats"],
            "search_train_curve": best_eval["search_log"],
            "final_train_curve": best_eval["final_log"],
        },
        "metrics_uad": best_eval["metrics_uad"],
    }


# ========================= Main =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_or_paths", default=None,
                        help=(
                            "uad_source: train_normal.npz,val_mixed.npz[,test_mixed.npz] | "
                            "adaptnas_combined: train_normal.npz,target_pool_unlabeled.npz,val_mixed.npz[,test_mixed.npz]"
                        ))
    parser.add_argument("--raw_smd_root", default="data/ServerMachineDataset",
                        help="Raw SMD root with train/test/(test_label|labels). Falls back to external SMD mirrors when needed.")
    parser.add_argument("--machine", default=None,
                        help="Machine id like machine-1-1. Used by family=omni_anomaly.")
    parser.add_argument("--swat_train_csv", default="data/SWaT/SWaT_Dataset_Normal_v1.csv",
                        help="Raw SWaT normal CSV. Used by family=usad.")
    parser.add_argument("--swat_test_csv", default="data/SWaT/SWaT_Dataset_Attack_v0.csv",
                        help="Raw SWaT attack CSV. Used by family=usad.")
    parser.add_argument("--epochs_pretrain", type=int, default=10)
    parser.add_argument("--search_candidates", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--combined_upper_gap", type=float, default=1.0,
                        help="Weight for source-target feature-gap term in unlabeled upper-level objective.")
    parser.add_argument(
        "--mode",
        type=str,
        default="adaptnas_combined",
        choices=["adaptnas_combined", "uad_source"],
        help=(
            "adaptnas_combined: UAD input, one-class-weighted target + bilevel AdaptNAS + final-only baselines.\n"
            "uad_source: UAD input, source-only (train_normal) one-class objective NAS; final one-class scoring."
        ),
    )
    parser.add_argument(
        "--family",
        type=str,
        default="default_nasade",
        choices=["default_nasade", "omni_anomaly", "usad", "tranad"],
        help=(
            "default_nasade: current CNN/Transformer-GRU-TCN + TS-TCC + pluggable one-class pipeline.\n"
            "omni_anomaly: paper-faithful OmniAnomaly family for raw SMD source-only runs.\n"
            "usad: paper-style USAD family for raw SWaT source-only runs.\n"
            "tranad: upstream-faithful TranAD family for raw SMD source-only runs."
        ),
    )
    parser.add_argument(
        "--oneclass_method",
        type=str,
        default="deepsvdd",
        choices=list_oneclass_methods(),
        help="One-class backend for default_nasade feature scoring/weighting.",
    )
    parser.add_argument("--oneclass_epochs", type=int, default=10)
    parser.add_argument("--oneclass_final_epochs", type=int, default=20)
    parser.add_argument("--oneclass_lr", type=float, default=1e-3)
    parser.add_argument("--oneclass_batch_size", type=int, default=1024)
    parser.add_argument("--oneclass_max_fit", type=int, default=5000)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--ocsvm_nu", type=float, default=0.05)
    parser.add_argument("--ocsvm_kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--ocsvm_gamma", type=str, default="scale")
    parser.add_argument("--ocsvm_degree", type=int, default=3)
    parser.add_argument("--ocsvm_coef0", type=float, default=0.0)
    parser.add_argument("--svdd_hidden_dim", type=int, default=128)
    parser.add_argument("--svdd_rep_dim", type=int, default=64)
    parser.add_argument("--svdd_nu", type=float, default=0.05)
    parser.add_argument("--svdd_warmup_epochs", type=int, default=2)
    parser.add_argument("--svdd_final_warmup_epochs", type=int, default=5)
    parser.add_argument("--ae_hidden_dim", type=int, default=128)
    parser.add_argument("--ae_latent_dim", type=int, default=64)
    parser.add_argument("--maha_hidden_dim", type=int, default=128)
    parser.add_argument("--maha_rep_dim", type=int, default=64)
    parser.add_argument("--maha_shrinkage", type=float, default=1e-2)
    parser.add_argument("--gmm_hidden_dim", type=int, default=128)
    parser.add_argument("--gmm_rep_dim", type=int, default=64)
    parser.add_argument("--gmm_components", type=int, default=3)
    parser.add_argument("--gmm_covariance_type", type=str, default="diag", choices=["diag", "full"])
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-4)
    parser.add_argument("--gmm_warmup_epochs", type=int, default=2)
    parser.add_argument("--proto_hidden_dim", type=int, default=128)
    parser.add_argument("--proto_rep_dim", type=int, default=64)
    parser.add_argument("--proto_count", type=int, default=4)
    parser.add_argument("--proto_separation_weight", type=float, default=0.1)
    parser.add_argument("--proto_separation_margin", type=float, default=1.0)
    parser.add_argument("--nas_search_iters", type=int, default=3)
    parser.add_argument("--combined_search_candidate_warmup_steps", type=int, default=50)
    parser.add_argument("--combined_search_steps", type=int, default=80)
    parser.add_argument("--combined_final_candidate_warmup_steps", type=int, default=80)
    parser.add_argument("--combined_final_steps", type=int, default=200)
    parser.add_argument("--combined_final_patience", type=int, default=10)
    parser.add_argument("--omni_epochs", type=int, default=20)
    parser.add_argument("--omni_final_epochs", type=int, default=20)
    parser.add_argument("--omni_lr", type=float, default=1e-3)
    parser.add_argument("--omni_patience", type=int, default=5)
    parser.add_argument("--omni_window_length", type=int, default=100)
    parser.add_argument("--omni_valid_ratio", type=float, default=0.3)
    parser.add_argument("--omni_batch_size", type=int, default=50)
    parser.add_argument("--omni_stride", type=int, default=1)
    parser.add_argument("--omni_test_n_z", type=int, default=1)
    parser.add_argument("--omni_search_iters", type=int, default=3)
    parser.add_argument("--omni_train_limit", type=int, default=0)
    parser.add_argument("--omni_test_limit", type=int, default=0)
    parser.add_argument("--omni_reference", type=str, default="paper", choices=["paper", "repo"])
    parser.add_argument("--omni_preprocess", type=str, default="official_minmax",
                        choices=["official_minmax", "train_zscore"])
    parser.add_argument("--omni_fixed_only", action="store_true")
    parser.add_argument("--omni_pot_q", type=float, default=0.0,
                        help="Override POT q. If <= 0, infer from --omni_reference.")
    parser.add_argument("--omni_pot_level", type=float, default=0.0,
                        help="Override POT level/low-quantile. If <= 0, infer from machine group and --omni_reference.")
    parser.add_argument("--usad_epochs", type=int, default=70)
    parser.add_argument("--usad_final_epochs", type=int, default=70)
    parser.add_argument("--usad_final_patience", type=int, default=0,
                        help="Patience for USAD final-stage early stopping. If <= 0, reuse --usad_patience.")
    parser.add_argument("--usad_lr", type=float, default=1e-3)
    parser.add_argument("--usad_patience", type=int, default=5)
    parser.add_argument("--usad_window_length", type=int, default=12)
    parser.add_argument("--usad_valid_ratio", type=float, default=0.2)
    parser.add_argument("--usad_batch_size", type=int, default=128)
    parser.add_argument("--usad_stride", type=int, default=1)
    parser.add_argument("--usad_downsample", type=int, default=5)
    parser.add_argument("--usad_latent_size", type=int, default=0,
                        help="Override paper-style z_size. If <= 0, use window_length * 100 like the upstream USAD SWaT notebook.")
    parser.add_argument("--usad_search_iters", type=int, default=3)
    parser.add_argument("--usad_train_limit", type=int, default=0)
    parser.add_argument("--usad_test_limit", type=int, default=0)
    parser.add_argument("--usad_score_alpha", type=float, default=0.5)
    parser.add_argument("--usad_score_beta", type=float, default=0.5)
    parser.add_argument("--usad_preprocess", type=str, default="train_minmax",
                        choices=["train_minmax", "train_zscore"])
    parser.add_argument("--usad_fixed_only", action="store_true")
    parser.add_argument("--usad_final_early_stopping", action="store_true")
    parser.add_argument("--usad_pot_q", type=float, default=1e-3)
    parser.add_argument("--usad_pot_level", type=float, default=0.99)
    parser.add_argument("--tranad_epochs", type=int, default=5)
    parser.add_argument("--tranad_final_epochs", type=int, default=5)
    parser.add_argument("--tranad_lr", type=float, default=1e-4)
    parser.add_argument("--tranad_patience", type=int, default=5)
    parser.add_argument("--tranad_window_length", type=int, default=10)
    parser.add_argument("--tranad_valid_ratio", type=float, default=0.2)
    parser.add_argument("--tranad_batch_size", type=int, default=128)
    parser.add_argument("--tranad_ff_dim", type=int, default=0,
                        help="Override fixed TranAD feedforward dim. If <= 0, use upstream default.")
    parser.add_argument("--tranad_dropout", type=float, default=0.1)
    parser.add_argument("--tranad_encoder_layers", type=int, default=0,
                        help="Override fixed TranAD encoder layer count. If <= 0, use upstream default.")
    parser.add_argument("--tranad_decoder_layers", type=int, default=0,
                        help="Override fixed TranAD decoder layer count. If <= 0, use upstream default.")
    parser.add_argument("--tranad_search_iters", type=int, default=2)
    parser.add_argument("--tranad_train_start", type=int, default=0)
    parser.add_argument("--tranad_train_limit", type=int, default=0)
    parser.add_argument("--tranad_test_start", type=int, default=0)
    parser.add_argument("--tranad_test_limit", type=int, default=0)
    parser.add_argument("--tranad_fixed_only", action="store_true")
    parser.add_argument("--tranad_pot_q", type=float, default=1e-3)
    parser.add_argument("--tranad_pot_level", type=float, default=0.99)
    args = parser.parse_args()

    set_global_seed(args.seed)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    device = args.device

    if args.family == "omni_anomaly":
        if args.mode != "uad_source":
            raise NotImplementedError("family=omni_anomaly is currently implemented only for mode=uad_source.")
        if not args.machine:
            raise ValueError("family=omni_anomaly requires --machine, e.g. --machine machine-1-1")

        print("[INFO] Family = omni_anomaly (paper-faithful source-only path).")
        res = run_omni_uad_source_family_raw(
            raw_smd_root=args.raw_smd_root,
            machine=args.machine,
            device=device,
            args=args,
        )
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("\n[OK] Saved outputs/results.json")
        return

    if args.family == "usad":
        if args.mode != "uad_source":
            raise NotImplementedError("family=usad is currently implemented only for mode=uad_source.")

        print("[INFO] Family = usad (paper-style source-only path on raw SWaT).")
        res = run_usad_uad_source_family_raw(
            swat_train_csv=args.swat_train_csv,
            swat_test_csv=args.swat_test_csv,
            device=device,
            args=args,
        )
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("\n[OK] Saved outputs/results.json")
        return

    if args.family == "tranad":
        if args.mode != "uad_source":
            raise NotImplementedError("family=tranad is currently implemented only for mode=uad_source.")
        if not args.machine:
            raise ValueError("family=tranad requires --machine, e.g. --machine machine-1-1")

        print("[INFO] Family = tranad (upstream-faithful source-only path on raw SMD).")
        res = run_tranad_uad_source_family_raw(
            raw_smd_root=args.raw_smd_root,
            machine=args.machine,
            device=device,
            args=args,
        )
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("\n[OK] Saved outputs/results.json")
        return

    if not args.dataset_or_paths:
        raise ValueError("--dataset_or_paths is required for family=default_nasade")

    def load_Xy(npz_path):
        X, y = load_npz_if_exists(npz_path)
        y = None if y is None else binarize_y(y)
        return X, y

    # -------- Load data --------
    parts = [p.strip() for p in args.dataset_or_paths.split(",")]
    X_train_norm, y_train_norm = (None, None)
    X_target_pool = None
    X_val, y_val = (None, None)
    X_test, y_test = (None, None)
    combined_has_separate_pool = False

    if args.mode == "uad_source":
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError("uad_source expects: train_normal.npz,val_mixed.npz[,test_mixed.npz]")

        X_train_norm, y_train_norm = load_Xy(parts[0])
        X_val, y_val = load_Xy(parts[1])
        if len(parts) == 3:
            X_test, y_test = load_Xy(parts[2])
    else:
        if len(parts) != 4:
            raise ValueError(
                "adaptnas_combined expects: "
                "train_normal.npz,target_pool_unlabeled.npz,val_mixed.npz,test_mixed.npz. "
                "This mode now requires a separate target_pool_unlabeled split to avoid label leakage."
            )

        X_train_norm, y_train_norm = load_Xy(parts[0])
        X_target_pool, _ = load_Xy(parts[1])
        X_val, y_val = load_Xy(parts[2])
        X_test, y_test = load_Xy(parts[3])
        combined_has_separate_pool = True

    if y_val is None:
        raise ValueError("val_mixed.npz must contain y for evaluation.")

    val_has_both_classes = np.unique(y_val).size >= 2
    test_has_both_classes = (y_test is not None) and (np.unique(y_test).size >= 2)

    print("[UAD INPUT]")
    print("  train_normal:", X_train_norm.shape, "y:", ("yes" if y_train_norm is not None else "no"))
    if X_target_pool is not None:
        print("  target_pool :", X_target_pool.shape, "y: no/ignored")
    print("  val_mixed   :", X_val.shape, "y:", ("yes" if y_val is not None else "no"))
    if X_test is not None:
        print("  test_mixed  :", X_test.shape, "y:", ("yes" if y_test is not None else "no"))

    if args.mode == "adaptnas_combined" and not val_has_both_classes:
        print(
            "[WARN] val_mixed contains only one class. Search no longer uses target labels, "
            "but validation metrics on val_mixed may be less informative."
        )

    print("[INFO] Normalizing window size to 128...")
    X_train_norm = fix_length(X_train_norm, window=128)
    if X_target_pool is not None:
        X_target_pool = fix_length(X_target_pool, window=128)
    X_val = fix_length(X_val, window=128)
    if X_test is not None:
        X_test = fix_length(X_test, window=128)

    if args.mode == "adaptnas_combined" and (X_target_pool is None or len(X_target_pool) == 0):
        raise ValueError("adaptnas_combined requires a non-empty target_pool_unlabeled split.")

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
    # pretrain data:
    # - combined: if cached SMD/SMAP entity path detected, can pretrain multi-entity; else pretrain on train_normal + target_pool
    # - uad_source: pretrain on train_normal only
    X_pretrain_multi = None
    pretrain_multi_name = None
    norm_path = parts[0].replace("\\", "/")
    candidate_root = infer_cached_pretrain_root(norm_path)
    if candidate_root and os.path.isdir(candidate_root):
        try:
            X_pretrain_multi = load_all_cached_entities_for_pretrain(candidate_root, window=128)
            pretrain_multi_name = os.path.basename(candidate_root)
        except RuntimeError:
            X_pretrain_multi = None
            pretrain_multi_name = None

    if args.mode == "uad_source":
        train_x = X_train_norm
        print(f"[INFO] TS-TCC pretraining on train_normal only: {train_x.shape[0]} windows.")
    else:
        if X_pretrain_multi is not None:
            train_x = X_pretrain_multi
            label = pretrain_multi_name if pretrain_multi_name is not None else "cached entities"
            print(f"[INFO] TS-TCC pretraining on multi-entity {label}: {train_x.shape[0]} windows.")
        else:
            pretrain_parts = [X_train_norm]
            pretrain_names = ["train_normal"]
            if X_target_pool is not None:
                pretrain_parts.append(X_target_pool)
                pretrain_names.append("target_pool_unlabeled")
            train_x = np.concatenate(pretrain_parts, axis=0)
            print(
                f"[INFO] TS-TCC pretraining on {' + '.join(pretrain_names)}: "
                f"{train_x.shape[0]} windows."
            )

    train_ss = {
        "samples": torch.tensor(train_x, dtype=torch.float32),
        "labels": torch.zeros(len(train_x)),
    }
    if len(train_x) < 2:
        raise ValueError(
            "TS-TCC pretraining requires at least 2 windows; "
            f"got {len(train_x)} from the current source/target setup."
        )

    config = Config()
    config.input_channels = in_ch
    if hasattr(config, "input_length"):
        config.input_length = 128
    if hasattr(config, "num_classes"):
        config.num_classes = num_classes
    ssl_batch_size = max(2, min(int(args.batch_size), int(len(train_x))))
    if ssl_batch_size != int(args.batch_size):
        print(
            f"[INFO] Adjusting TS-TCC self-supervised batch size "
            f"from {args.batch_size} to {ssl_batch_size} for {len(train_x)} windows."
        )
    config.batch_size = ssl_batch_size
    config.num_epoch = args.epochs_pretrain

    model = base_Model(config).to(device)
    temporal_contr_model = TC(config, device).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=3e-4)
    temp_opt = torch.optim.Adam(temporal_contr_model.parameters(), lr=config.lr, weight_decay=3e-4)
    tstcc = TSTrainer(model, temporal_contr_model, model_opt, temp_opt, device, config)

    train_dataset = Load_Dataset(train_ss, config, training_mode="self_supervised")
    train_loader = DataLoader(
        train_dataset,
        batch_size=ssl_batch_size,
        shuffle=True,
        drop_last=True,
    )
    tstcc.train(train_dl=train_loader, training_mode="self_supervised")

    tstcc_backbone = model
    tstcc_backbone.eval()

    # -------- Stage 2-4: Search --------
    N_ITERS = max(1, int(args.nas_search_iters))
    history = []
    best_arch = None
    search_oneclass_cfg = build_oneclass_config(
        args,
        epochs=args.oneclass_epochs,
        warmup_epochs=args.svdd_warmup_epochs,
    )
    final_oneclass_cfg = build_oneclass_config(
        args,
        epochs=args.oneclass_final_epochs,
        warmup_epochs=args.svdd_final_warmup_epochs,
    )
    oneclass_score_name = get_oneclass_score_name(args.oneclass_method)

    if args.mode == "uad_source":
        # split train_normal into train/val-normal for one-class objective selection
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(X_train_norm))
        n_val = max(50, int(0.2 * len(X_train_norm))) if len(X_train_norm) >= 250 else max(1, int(0.2 * len(X_train_norm)))
        idx_val = perm[:n_val]
        idx_tr = perm[n_val:] if n_val < len(X_train_norm) else perm[:max(1, len(X_train_norm)//2)]

        Xs_tr = X_train_norm[idx_tr]
        Xs_val = X_train_norm[idx_val]

        best_obj = float("inf")
        best_state = None

        for iter_id in range(N_ITERS):
            print(
                f"\n[ITER {iter_id+1}/{N_ITERS}] UAD_SOURCE search: "
                f"{args.oneclass_method} objective on train_normal..."
            )

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes=2).to(device)
                initialize_candidate_from_tstcc(cand, tstcc_backbone)

                obj, info = oneclass_objective_on_source_normal(
                    cand,
                    X_train_norm=Xs_tr,
                    X_val_norm=Xs_val,
                    device=device,
                    oneclass_cfg=search_oneclass_cfg,
                    seed=args.seed,
                )

                entry = {
                    "iter": iter_id + 1,
                    "arch": str(arch_c),
                    "oneclass_method": args.oneclass_method,
                    "oneclass_score_name": info["score_name"],
                    "oneclass_obj": float(obj),
                    "oneclass_mean_score": float(info["mean_score"]),
                }
                if args.oneclass_method == "deepsvdd":
                    entry["svdd_obj"] = float(obj)
                    entry["svdd_mean_dist2"] = float(info["mean_score"])
                history.append(entry)

                print(
                    f"  Candidate {i+1}/{args.search_candidates}: "
                    f"obj(mean_{info['score_name']})={obj:.6f}"
                )

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
        # ================= ADAPTNAS-COMBINED (source-normal + separate target pool when available) =================
        Xs_train, Xs_holdout = split_source_holdout_normal(X_train_norm, holdout_ratio=0.2, seed=args.seed)
        Xs = Xs_train
        Ys = Ys_source
        Xt_train = X_target_pool

        best_overall_upper_obj = float("inf")
        best_overall_arch = None
        best_overall_iter = None
        best_overall_cand_state = None

        tau = 1.0
        w_min = 0.05

        for iter_id in range(N_ITERS):
            print(
                f"\n[ITER {iter_id+1}/{N_ITERS}] ADAPTNAS_COMBINED: "
                f"{args.oneclass_method}-weighting + bilevel search..."
            )

            ds_source = ArrayDataset(Xs, Ys)
            best_iter_upper_obj = float("inf")
            best_arch_iter = None
            best_cand_iter = None

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes=2).to(device)
                initialize_candidate_from_tstcc(cand, tstcc_backbone)

                alpha_iter = 0.3 + 0.2 * iter_id

                # warmup on source normal-only labels (0) to stabilize feature extraction
                warmup_candidate_on_source(
                    cand,
                    ds_source,
                    device=device,
                    steps=args.combined_search_candidate_warmup_steps,
                    bs=args.batch_size,
                    lr=1e-3,
                )

                # Fit one-class backend on candidate features from source normal.
                _set_requires_grad(cand, False)
                Xs_fit = subsample_array(Xs, search_oneclass_cfg.max_fit, seed=args.seed)
                Fs = extract_candidate_features(cand, Xs_fit, device=device, batch_size=256)
                oneclass_backend = fit_oneclass_on_features(
                    Fs,
                    device=device,
                    oneclass_cfg=search_oneclass_cfg,
                    seed=args.seed,
                )

                raw = score_candidate_oneclass_stream(
                    cand,
                    oneclass_backend,
                    Xt_train,
                    device=device,
                    batch_size=256,
                )
                w_ent = robust_sigmoid_weights(raw, tau=tau, w_min=w_min)

                print(
                    f"    [W] w_ent stats: mean={w_ent.mean():.4f} "
                    f"min={w_ent.min():.4f} max={w_ent.max():.4f} "
                    f"| score={oneclass_backend.score_name}"
                )

                # bilevel training
                _set_requires_grad(cand, True)
                cand.train()

                ds_target = ArrayDataset(Xt_train, None, w=w_ent)
                upper_src_loader, upper_tgt_loader = build_upper_unlabeled_loaders(
                    Xs_holdout, Xt_train, w_ent, bs=args.batch_size
                )

                train_bilevel(
                    cand, ds_source, ds_target, None,
                    device=device,
                    steps=args.combined_search_steps,
                    bs=args.batch_size,
                    alpha=alpha_iter,
                    gamma=1.0,
                    lr_inner=1e-3,
                    lr_arch=1e-3,
                    use_cosine_decay=True,
                    early_stop=False,
                    upper_source_loader=upper_src_loader,
                    upper_target_loader=upper_tgt_loader,
                    upper_beta_gap=args.combined_upper_gap,
                )

                from src.adaptnas.optimizer import AdaptNASOptimizer
                opt = AdaptNASOptimizer(
                    cand, alpha=0.5, gamma=1.0,
                    lr_inner=1e-2, lr_arch=3e-3,
                    device=device
                )
                stats = opt.evaluate_upper_unlabeled(
                    upper_src_loader,
                    upper_tgt_loader,
                    alpha=alpha_iter,
                    beta_gap=args.combined_upper_gap,
                )
                upper_obj = float(stats["upper_obj"])

                entry = {
                    "iter": iter_id + 1,
                    "arch": str(arch_c),
                    "oneclass_method": args.oneclass_method,
                    "oneclass_score_name": oneclass_backend.score_name,
                    "upper_obj": float(upper_obj),
                    "src_obj": float(stats["src_obj"]),
                    "tgt_obj": float(stats["tgt_obj"]),
                    "gap_obj": float(stats["gap_obj"]),
                    "alpha": float(alpha_iter),
                    "tau": float(tau),
                    "w_min": float(w_min),
                }
                if args.oneclass_method == "deepsvdd":
                    entry["svdd_nu"] = float(args.svdd_nu)
                history.append(entry)

                print(
                    f"  Candidate {i+1}/{args.search_candidates}: upper_obj={upper_obj:.6f} "
                    f"(src={stats['src_obj']:.6f}, tgt={stats['tgt_obj']:.6f}, gap={stats['gap_obj']:.6f})"
                )

                if best_arch_iter is None or upper_obj < best_iter_upper_obj:
                    best_iter_upper_obj = float(upper_obj)
                    best_arch_iter = arch_c
                    best_cand_iter = cand

            print(f"[ITER {iter_id+1}] ✅ Best iter arch = {best_arch_iter} | upper_obj={best_iter_upper_obj:.6f}")

            if best_overall_arch is None or best_iter_upper_obj < best_overall_upper_obj:
                best_overall_upper_obj = float(best_iter_upper_obj)
                best_overall_arch = best_arch_iter
                best_overall_iter = iter_id + 1
                best_overall_cand_state = {k: v.detach().cpu().clone() for k, v in best_cand_iter.state_dict().items()}

        best_arch = best_overall_arch
        if best_arch is None or best_overall_cand_state is None:
            raise RuntimeError("adaptnas_combined: best_arch/state is None after search.")

        best_cand = CandidateModel(in_ch, best_arch, num_classes=2).to(device)
        best_cand.load_state_dict(best_overall_cand_state)

        print(f"\n[SEARCH DONE] ✅ Best OVERALL arch = {best_arch} (iter {best_overall_iter}) | best_upper_obj={best_overall_upper_obj:.6f}")

    # -------- Final stage --------
    print("[INFO] Final stage ...")

    res = {
        "mode": args.mode,
        "family": args.family,
        "best_arch": str(best_arch),
        "search_history": history,
        "eval_split": eval_name,
        "has_separate_target_pool": bool(combined_has_separate_pool),
        "val_mixed_has_both_classes": bool(val_has_both_classes),
        "oneclass": {
            "method": args.oneclass_method,
            "score_name": oneclass_score_name,
            "search_config": search_oneclass_cfg.to_dict(),
            "final_config": final_oneclass_cfg.to_dict(),
        },
        "search_objective": (
            f"source_normal_oneclass_compactness(mean_{oneclass_score_name}; method={args.oneclass_method})"
            if args.mode == "uad_source"
            else (
                "unlabeled_bilevel_upper_obj("
                "source_holdout_compactness + weighted_target_entropy + weighted_feature_gap; "
                f"target weights from {args.oneclass_method}:{oneclass_score_name})"
            )
        ),
    }

    if args.mode == "uad_source":
        # final: fit one-class backend on full train_normal; score eval split
        scores_eval, scores_train, final_backend_info = fit_final_oneclass_and_score(
            best_cand,
            X_train_norm=X_train_norm,
            X_eval=X_eval,
            device=device,
            oneclass_cfg=final_oneclass_cfg,
            seed=args.seed,
        )
        res["oneclass"]["final_backend"] = final_backend_info

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
        Xs, Xs_holdout = split_source_holdout_normal(X_train_norm, holdout_ratio=0.2, seed=args.seed)
        Ys = Ys_source
        Xt_train = X_target_pool

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
                X_target_pool=Xt_train,
                X_source_holdout=Xs_holdout,
                X_eval=X_eval, Y_eval=y_eval,
                args=args,
                device=device,
                in_ch=in_ch,
                N_ITERS=N_ITERS,
                tstcc_backbone=tstcc_backbone,
                seed=args.seed,
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
        res["oneclass"]["selection"] = "best_by_auroc_from_final_only_baselines"
        if best_by_auroc is not None and best_by_auroc.get("metrics_uad") is not None:
            res["metrics_uad"] = best_by_auroc["metrics_uad"]

    # -------- Save results --------
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("\n[OK] Saved outputs/results.json")


if __name__ == "__main__":
    main()
