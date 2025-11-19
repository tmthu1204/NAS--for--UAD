"""
Top-level orchestrator: preprocessing, TS-TCC pretrain, pseudo-labeling, AdaptNAS search, final training.
Implements AdaptNAS-Combined with TS-TCC pretraining and iterative refinement.
"""
import argparse
import os
import json
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
from src.utils.metrics import compute_ap_auroc, pot_threshold, f1_at_threshold, best_f1, event_f1_and_delay


# ============== Utility ==============
import numpy as np
import torch

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Py3.7+
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# === Weak & Strong augmentations ===
def jitter(x, sigma=0.01):
    return x + sigma * torch.randn_like(x)

def scale(x, low=0.8, high=1.2):
    factor = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(low, high)
    return x * factor

def permute(x, M=5):
    """Chia chuỗi thành M đoạn và xáo trộn."""
    B, C, T = x.shape
    perm_idx = torch.randperm(M)
    segs = torch.chunk(x, M, dim=2)
    return torch.cat([segs[i] for i in perm_idx], dim=2)

# === Contrastive collate function ===
def collate_fn(batch):
    xs, ys = [], []
    for item in batch:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            x, y = item
            xs.append(torch.tensor(x, dtype=torch.float32))
            ys.append(torch.tensor(y, dtype=torch.long))
        else:
            xs.append(torch.tensor(item, dtype=torch.float32))

    data = torch.stack(xs)                  # [B, T, C]
    data = data.permute(0, 2, 1)            # [B, C, T]
    labels = torch.stack(ys) if ys else torch.zeros(data.size(0), dtype=torch.long)

    # === weak vs strong augmentations ===
    aug_w = jitter(scale(data.clone(), 0.8, 1.2), sigma=0.01)
    aug_s = permute(jitter(data.clone(), sigma=0.01), M=5)

    return data, labels, aug_w, aug_s





def pseudo_label_balanced(clf, Zt, top_p=0.2, n_classes=2):
    """
    Chọn pseudo-labels cân bằng lớp: mỗi lớp lấy ~k_per mẫu tự tin nhất.
    """
    probs = clf.predict_proba(Zt)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    k_total = max(1, int(top_p * len(Zt)))
    k_per = max(1, k_total // n_classes)

    keep_idx = []
    for c in range(n_classes):
        idx_c = np.where(pred == c)[0]
        if idx_c.size == 0:
            continue
        sel = idx_c[np.argsort(-conf[idx_c])[:k_per]]
        keep_idx.append(sel)
    if len(keep_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), conf
    keep_idx = np.concatenate(keep_idx)
    return keep_idx, pred[keep_idx], conf

def filter_by_consistency(clf, trainer, Xt, device, keep_idx, thr=0.0, batch_size=256):
    """
    Giữ mẫu mà dự đoán trên 2 view (weak/strong) trùng nhau.
    Nếu muốn, có thể yêu cầu thêm 'conf >= thr'.
    """
    # Tạo 2 view nhanh bằng jitter/permute tại tensor level
    def _augment_np(X):
        Xt = torch.tensor(X, dtype=torch.float32).permute(0,2,1)   # [B,C,T]
        aug_w = (Xt + 0.01*torch.randn_like(Xt)).permute(0,2,1).numpy()         # jitter nhẹ
        # strong: permute time
        B,T,C = X.shape
        M = 5
        out = []
        for i in range(B):
            segs = np.array_split(np.arange(T), M)
            np.random.shuffle(segs)
            warp = np.concatenate(segs)
            out.append(X[i, warp, :])
        aug_s = np.stack(out, axis=0)
        return aug_w, aug_s

    X_keep = Xt[keep_idx]
    aug_w, aug_s = _augment_np(X_keep)

    # extract features cho 2 view
    Zw = extract_features(trainer, aug_w, device, batch_size)
    Zs = extract_features(trainer, aug_s, device, batch_size)

    pw = clf.predict_proba(Zw); ps = clf.predict_proba(Zs)
    yw = pw.argmax(1); ys = ps.argmax(1)
    cw = pw.max(1);    cs = ps.max(1)

    agree = (yw == ys)
    if thr > 0:
        conf_ok = (0.5*(cw+cs) >= thr)
        agree = agree & conf_ok

    new_idx = keep_idx[agree]
    new_y   = yw[agree]
    new_c   = 0.5*(cw[agree]+cs[agree])
    return new_idx, new_y, new_c



def load_npz_if_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    if 'X' not in data:
        raise ValueError(f"{path} missing key 'X'")
    X = data['X']
    y = data['y'] if 'y' in data else None
    return X, y


def build_validation(ds_source, ds_target, beta=0.5, m=200, bs=64, seed=42, fixed_s_idx=None):
    """
    Build 3 DataLoader:
      - val_src: chỉ source
      - val_tgt: chỉ target
      - val_hybrid: gộp S/T theo tỉ lệ beta (để giữ compat cũ)
    Trả về: (val_hybrid, s_idx, val_src, val_tgt, beta_eff)
    """
    rng = np.random.RandomState(seed)
    ns, nt = len(ds_source), len(ds_target)
    ms = max(1, int(beta * m))
    mt = max(1, m - ms)

    # source index cố định từ vòng 1
    if fixed_s_idx is None:
        s_idx = rng.choice(ns, size=min(ms, ns), replace=False)
    else:
        s_idx = fixed_s_idx[:min(ms, len(fixed_s_idx))]

    # target chọn ổn định theo seed
    t_all = np.arange(nt)
    rng.shuffle(t_all)
    t_idx = t_all[:min(mt, nt)]

    Xs_val = np.stack([ds_source[i][0].numpy() for i in s_idx])
    ys_val = np.array([ds_source[i][1].item() for i in s_idx])

    Xt_val = np.stack([ds_target[i][0].numpy() for i in t_idx])
    yt_val = np.array([ds_target[i][1].item() for i in t_idx])

    val_src = DataLoader(ArrayDataset(Xs_val, ys_val), batch_size=bs)
    val_tgt = DataLoader(ArrayDataset(Xt_val, yt_val), batch_size=bs)

    # hybrid (để tương thích các chỗ còn dùng val_loader)
    Xhyb = np.concatenate([Xs_val, Xt_val], axis=0)
    yhyb = np.concatenate([ys_val, yt_val], axis=0)
    val_hybrid = DataLoader(ArrayDataset(Xhyb, yhyb), batch_size=bs)

    beta_eff = len(ys_val) / max(1, len(yhyb))
    return val_hybrid, s_idx, val_src, val_tgt, beta_eff


# ============== Model wrapper ==============
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class CandidateModel(torch.nn.Module):
    def __init__(self, in_ch, arch, num_classes=2):
        super().__init__()

        self.seq_type = arch.seq_type

        # Encoder CNN dùng dilations + activation từ ArchConfig
        self.encoder = EncoderCNN(
            in_ch,
            arch.enc_filters,
            arch.enc_kernels,
            arch.enc_strides,
            pool=arch.enc_pool,
            activation=arch.enc_activation,
            dilations=arch.enc_dilations
        )

        last = arch.enc_filters[-1]
        self.to_d = nn.Conv1d(last, arch.d_model, kernel_size=1)

        # Sequence module
        if self.seq_type == "transformer":
            self.sequence = ARTransformer(
                d_model=arch.d_model,
                nhead=arch.seq_heads,
                num_layers=arch.seq_layers,
                dim_feedforward=arch.seq_hidden
            )
        elif self.seq_type == "gru":
            # dùng luôn d_model làm hidden_size cho gọn
            self.sequence = nn.GRU(
                input_size=arch.d_model,
                hidden_size=arch.d_model,
                num_layers=arch.seq_layers,
                batch_first=True,
                bidirectional=False
            )
        elif self.seq_type == "tcn":
            blocks = []
            in_c = arch.d_model
            for l in range(arch.seq_layers):
                dilation = arch.seq_dilation ** l
                blocks.append(
                    TCNBlock(
                        in_c,
                        arch.d_model,
                        kernel_size=arch.seq_kernel,
                        dilation=dilation
                    )
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
        # EncoderCNN trả (B, T, C_enc)
        z = self.encoder(x)          # (B, T, C_enc)
        z = z.transpose(1, 2)        # (B, C_enc, T)
        z = self.to_d(z)             # (B, d_model, T)

        if self.seq_type == "tcn":
            h = self.sequence(z)     # (B, d_model, T)
            f = h.mean(dim=2)        # GAP theo thời gian -> (B, d_model)
        else:
            z_seq = z.transpose(1, 2)  # (B, T, d_model)
            if self.seq_type == "transformer":
                out = self.sequence(z_seq)   # (B, T, d_model)
                f = out.mean(dim=1)
            elif self.seq_type == "gru":
                _, h_n = self.sequence(z_seq)  # h_n: (num_layers, B, d_model)
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




# ============== TS-TCC feature extractor ==============
def extract_features(trainer, X, device, batch_size=256):
    """
    Trích đặc trưng từ encoder TS-TCC: average pooling theo TRỤC THỜI GIAN.
    Đầu ra: [N, C] để huấn luyện bộ phân lớp (LR) ổn định hơn.
    """
    trainer.model.eval()
    feats = []
    dl = DataLoader(ArrayDataset(X), batch_size=batch_size)
    with torch.no_grad():
        for xb in dl:
            xb = xb.permute(0, 2, 1).to(device)   # [B, C, T]
            _, z = trainer.model(xb)              # z: [B, C, T]
            f = z.mean(dim=2)                     # ✅ GAP theo THỜI GIAN -> [B, C]
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)



def fix_length(X, window=128):
    """Pad hoặc cắt từng mẫu cho cùng độ dài window."""
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
    """
    Gom dữ liệu của tất cả máy trong SMD để pretrain TS-TCC:
      - dùng cả source.npz (normal) và target.npz (mixed)
      - chỉ dùng X, không dùng y
    """
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




# ============== Main Pipeline ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_or_paths', required=True)
    parser.add_argument('--epochs_pretrain', type=int, default=10)
    parser.add_argument('--search_candidates', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--uad', action='store_true',
                    help='Enable UAD mode: source=train_normal (normal only), target=val_mixed (unlabeled in training), validation uses val_mixed labels.')
    args = parser.parse_args()

    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/checkpoints', exist_ok=True)

    ds = args.dataset_or_paths
    device = args.device

    def load_Xy(npz_path):
        X, y = load_npz_if_exists(npz_path)
        return X, (None if y is None else y.astype(int))

    if args.uad:
        # Hỗ trợ "3 file": train_normal.npz, val_mixed.npz, test_mixed.npz
        parts = [p.strip() for p in ds.split(',')]
        if len(parts) < 2:
            raise ValueError("UAD mode expects at least 2 paths: train_normal.npz,val_mixed.npz[,test_mixed.npz]")

        Xs, _         = load_Xy(parts[0])       # normal only (no labels)
        Xval, Yval    = load_Xy(parts[1])       # mixed with labels
        Xtest, Ytest  = (None, None)
        if len(parts) >= 3:
            Xtest, Ytest = load_Xy(parts[2])    # optional test set with labels

        print("UAD mode:")
        print("  train_normal:", Xs.shape)
        print("  val_mixed   :", Xval.shape, "(labels:", (Yval is not None), ")")
        if Xtest is not None:
            print("  test_mixed  :", Xtest.shape, "(labels:", (Ytest is not None), ")")

        # Chuẩn hóa độ dài
        print("[INFO] Normalizing window size to 128...")
        Xs   = fix_length(Xs,   window=128)
        Xval = fix_length(Xval, window=128)
        if Xtest is not None:
            Xtest = fix_length(Xtest, window=128)

        in_ch = Xs.shape[-1]
        num_classes = 2  # anomaly detection: normal vs anomaly

    else:
        # ==== NHÁNH CŨ (HAR / sleepEDF / SMD DA) ====
        ds_low = ds.lower()
        X_pretrain_multi = None  # mặc định: không dùng multi-machine

        if ',' in ds_low:
            s_path, t_path = ds_low.split(',', 1)
            s_path = s_path.strip()
            t_path = t_path.strip()
            Xs, Ys = load_npz_if_exists(s_path)
            Xt, Yt = load_npz_if_exists(t_path)
            Ys = Ys.astype(int)
            Yt = Yt.astype(int) if Yt is not None else None

            # nếu là SMD (data/smd/machine-*/source.npz) thì bật multi-machine pretrain
            if "data/smd" in s_path and "machine-" in s_path:
                machine_dir = os.path.dirname(s_path)      # .../data/smd/machine-1-1
                smd_root    = os.path.dirname(machine_dir) # .../data/smd
                X_pretrain_multi = load_all_smd_for_pretrain(smd_root, window=128)


        elif ds_low == 'uci_har':
            Xs, Ys = load_npz_if_exists('data/uci_har/source.npz')
            Xt, Yt = load_npz_if_exists('data/uci_har/target.npz')
        elif ds_low in ('sleepedf', 'sleep_edf', 'sleep-edf'):
            Xs, Ys = load_npz_if_exists('data/sleepedf/source.npz')
            Xt, Yt = load_npz_if_exists('data/sleepedf/target.npz')
        else:
            raise ValueError("Unknown dataset name.")

        print("Unique labels (source):", np.unique(Ys))
        if Yt is not None:
            print("Unique labels (target):", np.unique(Yt))

        print("[INFO] Normalizing window size to 128...")
        Xs = fix_length(Xs, window=128)
        Xt = fix_length(Xt, window=128)
        in_ch = Xs.shape[-1]

        # Suy số lớp từ cả source và target, và tối thiểu 2 lớp
        if Yt is not None:
            num_classes = int(max(Ys.max(), Yt.max())) + 1
        else:
            num_classes = int(Ys.max()) + 1
        num_classes = max(num_classes, 2)

        print(f"[INFO] Detected num_classes = {num_classes} (in_ch={in_ch})")




    # === Stage 1: TS-TCC pretraining (self-supervised) ===
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
    temp_opt  = torch.optim.Adam(temporal_contr_model.parameters(), lr=config.lr, weight_decay=3e-4)
    tstcc = TSTrainer(model, temporal_contr_model, model_opt, temp_opt, device, config)

    if args.uad:
        # chỉ normal của 1 domain UAD cũ
        train_x = Xs
    else:
        # nếu có multi-machine pretrain cho SMD thì dùng toàn bộ X từ tất cả machine,
        # ngược lại thì gộp source+target của domain hiện tại như cũ
        if 'X_pretrain_multi' in locals() and X_pretrain_multi is not None:
            train_x = X_pretrain_multi
            print(f"[INFO] TS-TCC pretraining on multi-machine SMD: {train_x.shape[0]} windows.")
        else:
            train_x = np.concatenate([Xs, Xt], axis=0)
            print(f"[INFO] TS-TCC pretraining on current domain only: {train_x.shape[0]} windows.")

    train_ss = {
        "samples": torch.tensor(train_x, dtype=torch.float32),
        "labels":  torch.zeros(len(train_x))
    }

    train_dataset = Load_Dataset(train_ss, config, training_mode="self_supervised")
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    tstcc.train(train_dl=train_loader, training_mode="self_supervised")



    # === Stage 2–4: AdaptNAS iterative search ===
    N_ITERS = 3
    history = []
    fixed_s_idx = None

    if args.uad:
        # source: normal có nhãn 0 (để loss_s có điểm tựa); target: unlabeled (val_mixed.X)
        Ys = np.zeros(len(Xs), dtype=int)
        Xtu = Xval.copy()     # unlabeled target for training
        Yval = Yval.astype(int) if Yval is not None else None

        ds_source = ArrayDataset(Xs, Ys)                 # (x,y)
        dl_val    = DataLoader(ArrayDataset(Xval, Yval), batch_size=args.batch_size, shuffle=False)

        # val_src: dùng một phần train_normal (nhãn 0) làm "source val" cho đúng form AdaptNAS
        n_src_val = min(len(Xs), 200)
        idx_src_val = np.random.RandomState(42).choice(len(Xs), size=n_src_val, replace=False)
        val_src = DataLoader(ArrayDataset(Xs[idx_src_val], np.zeros(n_src_val, dtype=int)),
                            batch_size=args.batch_size, shuffle=False)
        val_tgt = dl_val  # target val có nhãn từ val_mixed


        best_arch = None
        for iter_id in range(N_ITERS):
            print(f"\n[ITER {iter_id+1}/{N_ITERS}] AdaptNAS search (UAD)...")

            # unlabeled target dataset
            ds_target_u = ArrayDataset(Xtu)              # (x) only

            best_val_acc = -float("inf")
            best_cand = None
            best_arch = None

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes=2).to(device)

                alpha_iter = 0.3 + 0.2 * iter_id
                # dùng dl_val trực tiếp (có nhãn) để step_upper/earlystop
                train_log = train_bilevel(
                    cand, ds_source, ds_target_u, dl_val,
                    device=device, steps=50, bs=args.batch_size,
                    alpha=alpha_iter, gamma=1.0,
                    use_cosine_decay=True, early_stop=False
                )

                from src.adaptnas.optimizer import AdaptNASOptimizer
                opt = AdaptNASOptimizer(cand, alpha=0.5, gamma=1.0, lr_inner=1e-3, lr_arch=1e-3, device=device)
                stats = opt.step_upper_combined(val_src, val_tgt, alpha=alpha_iter)
                val_acc = 1.0 - stats["hybrid_err"]
                acc_quick = quick_validate(cand, val_tgt, device)  # acc trên target val

                history.append({
                    'iter': iter_id + 1,
                    'arch': str(arch_c),
                    'acc_quick': acc_quick,
                    'val_acc':   val_acc,
                    'src_err':   stats["src_err"],
                    'tgt_err':   stats["tgt_err"],
                    'hybrid_err':stats["hybrid_err"],
                    'alpha':     alpha_iter
                })
                print(f"  Candidate {i+1}/{args.search_candidates}: "
                    f"val_acc={val_acc:.4f} | src_err={stats['src_err']:.4f} | tgt_err={stats['tgt_err']:.4f}")


                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_arch = arch_c
                    best_cand = cand

            print(f"[ITER {iter_id+1}] ✅ Best arch = {best_arch}")
            print("[INFO] Selected architecture summary:")
            print(best_cand)

    else:
        N_ITERS = 3
        history, best_arch = [], None

        fixed_s_idx = None  # ✅ sẽ nhận từ lần build đầu và tái sử dụng về sau


        for iter_id in range(N_ITERS):
            print(f"\n[ITER {iter_id+1}/{N_ITERS}] Pseudo-labeling & AdaptNAS search...")

            # --- extract features from pretrained TS-TCC ---
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC
            from sklearn.ensemble import IsolationForest
            from sklearn.calibration import CalibratedClassifierCV


            Zs = extract_features(tstcc, Xs, device)
            Zt = extract_features(tstcc, Xt, device)

            # ===== SEED POSITIVE BẰNG ISOLATION FOREST =====
            if np.unique(Ys).size < 2:
                # train IF trên normal Zs
                iso = IsolationForest(
                    n_estimators=200,
                    contamination=0.1,   # 10% anomaly giả định
                    random_state=42
                )
                iso.fit(Zs)
                # score_samples: càng nhỏ càng bất thường → đổi dấu cho dễ hiểu
                scores_t = -iso.score_samples(Zt)  # [Nt]

                top_p = 0.10
                k_pos = max(1, int(top_p * len(scores_t)))
                pos_idx_seed = np.argsort(-scores_t)[:k_pos]

                Z_seed = np.concatenate([Zs, Zt[pos_idx_seed]], axis=0)
                y_seed = np.concatenate([
                    np.zeros(len(Zs), dtype=int),
                    np.ones(len(pos_idx_seed), dtype=int)
                ], axis=0)
            else:
                Z_seed, y_seed = Zs, Ys

            # ===== CLASSIFIER: STANDARD SCALER + LINEAR SVM (CALIBRATED) =====
            base_svm = LinearSVC(
                class_weight='balanced',
                max_iter=5000
            )
            svm_calib = CalibratedClassifierCV(base_svm, cv=3)

            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", svm_calib)
            ])
            clf.fit(Z_seed, y_seed)


            # === phần còn lại giữ nguyên ===
            # Chọn pseudo-balanced theo lớp
            idx, yps, pseudo_confidences = pseudo_label_balanced(clf, Zt, top_p=0.2, n_classes=num_classes)

            # Siết điều kiện nhất quán giữa các view
            idx, yps, pseudo_confidences = filter_by_consistency(
                clf, tstcc, Xt, device, idx, thr=0.8, batch_size=256
            )
            # Nếu sau filter không còn mẫu nào, hạ tiêu chí và/hoặc chuyển sang unlabeled target
            if idx.size == 0:
                print("[WARN] No pseudo-labeled target samples after consistency filtering.")
                # 1) Thử nới lỏng: bỏ consistency, lấy top-k theo confidence
                probs_t = clf.predict_proba(extract_features(tstcc, Xt, device, 256))
                conf_t = probs_t.max(axis=1)
                k = max(1, int(0.05 * len(Xt)))  # lấy top 5% tối thiểu 1 mẫu
                idx_top = conf_t.argsort()[::-1][:k]
                yps_top = probs_t[idx_top].argmax(1)

                if k > 0:
                    idx, yps = idx_top, yps_top
                    pseudo_confidences = conf_t  # giữ full để gán w nếu cần
                    print(f"[INFO] Fallback to top-{k} confident target samples without consistency.")
                else:
                    # 2) Cuối cùng: dùng toàn bộ Xt ở chế độ unlabeled (entropy minimization)
                    print("[INFO] Fallback to unlabeled target training (entropy minimization).")
                    ds_target = ArrayDataset(Xt)  # chỉ (x) -> trainer sẽ dùng entropy ở nhánh yt is None

            w_pseudo = pseudo_confidences



            # (khuyến nghị) lưu cả weights
            np.save(f'outputs/pseudo_idx_iter{iter_id+1}.npy', idx)
            np.save(f'outputs/pseudo_y_iter{iter_id+1}.npy', yps)
            np.save(f'outputs/pseudo_conf_iter{iter_id+1}.npy', pseudo_confidences)

            ds_source = ArrayDataset(Xs, Ys)
            ds_target = ArrayDataset(Xt[idx], yps, w=w_pseudo)   # ✅ đúng
            if iter_id == 0:
                # lần đầu: fixed_s_idx=None -> hàm trả về s_idx để lưu
                val_loader, fixed_s_idx, val_src, val_tgt, beta_eff = build_validation(
                    ds_source, ds_target, bs=args.batch_size, seed=42, fixed_s_idx=None
                )
            else:
                # các vòng sau: tái dùng s_idx đã cố định từ vòng 1
                val_loader, _, val_src, val_tgt, beta_eff = build_validation(
                    ds_source, ds_target, bs=args.batch_size, seed=42, fixed_s_idx=fixed_s_idx
                )


            # --- Search phase: multiple random candidate architectures ---
            best_val_acc = -float("inf")   # ✅ KHỞI TẠO Ở ĐÂY (trước vòng for)
            best_cand = None
            best_arch = None

            for i in range(args.search_candidates):
                arch_c = sample_arch()
                cand = CandidateModel(in_ch, arch_c, num_classes).to(device)

                alpha_iter = 0.3 + 0.2 * iter_id   # 0.3 -> 0.5 -> 0.7

                # lower-level: nhiều bước hơn, lr_inner lớn hơn (tinh thần batch 512, lr 0.25)
                train_bilevel(
                    cand, ds_source, ds_target, val_loader,
                    device=device,
                    steps=80,              # thay vì 150
                    bs=args.batch_size,    # nhưng ta sẽ chạy bs 128 thôi
                    alpha=alpha_iter,
                    gamma=1.0,
                    lr_inner=1e-3,         # quay về 1e-3
                    lr_arch=1e-3,
                    use_cosine_decay=True,
                    early_stop=False
                )


                from src.adaptnas.optimizer import AdaptNASOptimizer
                opt = AdaptNASOptimizer(
                    cand,
                    alpha=0.5,
                    gamma=1.0,
                    lr_inner=1e-2,                # match với train_bilevel
                    lr_arch=3e-3,
                    device=device
                )
                stats = opt.step_upper_combined(val_src, val_tgt, alpha=alpha_iter)
                val_acc   = 1.0 - stats["hybrid_err"]
                acc_quick = quick_validate(cand, val_loader, device)

                history.append({
                    'iter': iter_id + 1,
                    'arch': str(arch_c),
                    'acc_quick': acc_quick,
                    'val_acc':   val_acc,
                    'src_err':   stats["src_err"],
                    'tgt_err':   stats["tgt_err"],
                    'hybrid_err':stats["hybrid_err"],
                    'alpha':     alpha_iter,
                    'beta':      beta_eff
                })
                print(f"  Candidate {i+1}/{args.search_candidates}: "
                    f"val_acc={val_acc:.4f} | src_err={stats['src_err']:.4f} | tgt_err={stats['tgt_err']:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_arch = arch_c
                    best_cand = cand


                stats = opt.step_upper_combined(val_src, val_tgt, alpha=alpha_iter)
                val_acc   = 1.0 - stats["hybrid_err"]
                acc_quick = quick_validate(cand, val_loader, device)

                history.append({
                    'iter': iter_id + 1,
                    'arch': str(arch_c),
                    'acc_quick': acc_quick,
                    'val_acc':   val_acc,
                    'src_err':   stats["src_err"],
                    'tgt_err':   stats["tgt_err"],
                    'hybrid_err':stats["hybrid_err"],
                    'alpha':     alpha_iter,
                    'beta':      beta_eff
                })
                print(f"  Candidate {i+1}/{args.search_candidates}: "
                    f"val_acc={val_acc:.4f} | src_err={stats['src_err']:.4f} | tgt_err={stats['tgt_err']:.4f}")


                # ✅ chọn theo val_acc cao nhất
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_arch = arch_c
                    best_cand = cand


            print(f"[ITER {iter_id+1}] ✅ Best arch = {best_arch}")

            # === After AdaptNAS search finishes ===
            print(f"[INFO] Selected architecture (A*) summary after iteration {iter_id+1}:")
            print(best_cand)  # ✅ dùng best_cand, không phải cand
        pass

        # Nếu cand có arch_params
        if hasattr(best_cand, "arch_params"):
            with torch.no_grad():
                probs = torch.softmax(best_cand.arch_params, dim=0)
                print("[INFO] Architecture operation probabilities:")
                for i, p in enumerate(probs):
                    print(f"  Op {i}: {p.item():.4f}")

        # === Evaluate pseudo-labels quality ===
        print(f"[INFO] Pseudo-labels generated: {len(yps)} samples")
        conf_mean = np.mean(pseudo_confidences)
        print(f"[INFO] Average pseudo-label confidence: {conf_mean:.4f}")

        # Nếu có ground truth của source validation
        if "Ys_val" in locals() and "classifier" in locals():
            from sklearn.metrics import accuracy_score
            pseudo_acc = accuracy_score(Ys_val, classifier.predict(Zs_val))
            print(f"[INFO] Estimated pseudo-label accuracy on source validation: {pseudo_acc*100:.2f}%")

    # === Final training on best architecture ===
    print("[INFO] Final training with best architecture...")

    if args.uad:
        final_model = CandidateModel(in_ch, best_arch, num_classes).to(device)
        ds_source = ArrayDataset(Xs, Ys)
        Xt_pseudo = Xt[idx]
        ds_target = ArrayDataset(Xt_pseudo, yps)
        val_loader, _, _, _, _ = build_validation(
            ds_source, ds_target, bs=args.batch_size, seed=42, fixed_s_idx=fixed_s_idx
        )

        alpha_final = 0.3 + 0.2 * (N_ITERS - 1)  # N_ITERS=3 -> 0.7

        train_log = train_bilevel(
            final_model, ds_source, ds_target, val_loader,
            device=device,
            steps=400,               # trước 150, giờ train dài hơn (tương đương epochs=250)
            bs=args.batch_size,      # sẽ set =512 khi chạy
            alpha=alpha_final,
            gamma=1.0,
            lr_inner=1e-2,           # match với search
            lr_arch=3e-3,
            use_cosine_decay=True,
            early_stop=True,
            patience=10,
            ckpt_path="outputs/checkpoints/final_best.pt"
        )


        # === Evaluation on test_mixed (nếu có) ===
        metrics, roc_data = None, None
        if 'Xtest' in locals() and Xtest is not None and Ytest is not None:
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

            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
            from sklearn.preprocessing import label_binarize
            prec, rec, f1, _ = precision_recall_fscore_support(Ytest, y_pred, average='macro', zero_division=0)
            Yt_bin = label_binarize(Ytest, classes=[0,1])
            try:
                auc_score = roc_auc_score(Yt_bin, probs, average='macro', multi_class='ovr')
            except ValueError:
                auc_score = 0.5
            metrics = {'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_score}

            fpr, tpr, _ = roc_curve(Yt_bin[:,1], probs[:,1])
            roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    else:
        print("[INFO] Final training with best architecture...")

        final_model = CandidateModel(in_ch, best_arch, num_classes).to(device)
        ds_source = ArrayDataset(Xs, Ys)
        Xt_pseudo = Xt[idx]
        ds_target = ArrayDataset(Xt_pseudo, yps)

        # build_validation trả về 5 giá trị
        val_loader, _, _, _, _ = build_validation(
            ds_source, ds_target, bs=args.batch_size, seed=42, fixed_s_idx=fixed_s_idx
        )


        alpha_final = 0.3 + 0.2 * (N_ITERS - 1)  # ví dụ N_ITERS=3 -> 0.7
        train_log = train_bilevel(
            final_model, ds_source, ds_target, val_loader,
            device=device,
            steps=200,             # đủ dài nhưng không quá
            bs=args.batch_size,
            alpha=alpha_final,
            gamma=1.0,
            lr_inner=1e-3,
            lr_arch=1e-3,
            use_cosine_decay=True,
            early_stop=True,
            patience=10,
            ckpt_path="outputs/checkpoints/final_best.pt"
        )








        # === Evaluation on target ===c
        metrics, roc_data = None, None
        if Yt is not None:
            final_model.eval()
            all_probs = []
            dl_full = DataLoader(ArrayDataset(Xt), batch_size=256)
            with torch.no_grad():
                for xb in dl_full:
                    xb = xb.to(device)
                    logits, _ = final_model(xb)
                    all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs = np.concatenate(all_probs, axis=0)
            y_pred = probs.argmax(axis=1)

            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
            from sklearn.preprocessing import label_binarize
            prec, rec, f1, _ = precision_recall_fscore_support(
                Yt, y_pred, average='macro', zero_division=0
            )

            # ====== MULTI-CLASS ROC/AUC (chỉ dùng khi >2 lớp) ======
            if probs.shape[1] > 2:
                Yt_bin = label_binarize(Yt, classes=np.arange(probs.shape[1]))
                try:
                    auc_score = roc_auc_score(Yt_bin, probs, average='macro', multi_class='ovr')
                except ValueError:
                    auc_score = 0.5
                metrics = {'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_score}

                fpr, tpr = {}, {}
                for i in range(probs.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(Yt_bin[:, i], probs[:, i])
                roc_data = {
                    'fpr': {i: fpr[i].tolist() for i in fpr},
                    'tpr': {i: tpr[i].tolist() for i in tpr}
                }
            else:
                # Binary case (SMD / UAD): AUC & ROC sẽ được tính ở block UAD metrics phía dưới
                metrics = {'precision': prec, 'recall': rec, 'f1': f1}
                roc_data = None

        pass

    

        # ==== COMMON EVAL (UAD & DA) ====
        # Với UAD (old branch): đánh giá trên (Xval, Yval).
        # Với DA (new UAD on SMD): đánh giá trên (Xt, Yt).
        paper_metrics = {}
        uad_metrics = None

        if args.uad:
            eval_X, eval_y = Xval, Yval
        else:
            eval_X, eval_y = (Xt if 'Xt' in locals() else None), (Yt if 'Yt' in locals() else None)

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
            pred  = probs.argmax(axis=1)

            # Paper-style (classification)
            target_acc = float((pred == eval_y).mean())
            target_err = 100.0 * (1.0 - target_acc)
            paper_metrics = {'target_top1_acc': target_acc, 'target_error_percent': target_err}

            # === UAD metrics cho case binary anomaly (SMD hoặc nhánh --uad cũ) ===
            # Điều kiện: mô hình có đúng 2 lớp (normal/anomaly)
            if probs.shape[1] == 2:
                scores = probs[:, 1]  # xác suất anomaly
                ap, auroc = compute_ap_auroc(eval_y, scores)

                # Học ngưỡng POT từ train_normal (Xs)
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
                    'ap': ap, 'auroc': auroc,
                    'f1_pot': f1_pot, 'precision_pot': p_pot, 'recall_pot': r_pot, 'thr_pot': thr_pot,
                    'f1_best': f1_best, 'precision_best': p_best, 'recall_best': r_best, 'thr_best': thr_best,
                    'event_f1': ev['event_f1'], 'event_precision': ev['event_precision'], 'event_recall': ev['event_recall'],
                    'delay_mean': ev['delay_mean'], 'delay_median': ev['delay_median']
                }


        res = {'best_arch': str(best_arch), 'search_history': history}
        if paper_metrics:
            res['metrics_paper'] = paper_metrics
        if uad_metrics is not None:
            res['metrics_uad'] = uad_metrics
        if 'metrics' in locals() and metrics is not None:
            res['metrics'] = metrics
        if 'roc_data' in locals() and roc_data is not None:
            res['roc'] = roc_data
        if 'train_log' in locals():
            res['train_curves'] = train_log

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/results.json", "w") as f:
            json.dump(res, f, indent=2)



if __name__ == '__main__':
    main()
