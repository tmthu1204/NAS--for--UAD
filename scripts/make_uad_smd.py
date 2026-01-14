# scripts/make_uad_smd.py
import argparse
import os
import numpy as np


def load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    if "X" not in data:
        raise ValueError(f"{path} missing key 'X'")
    X = data["X"]
    y = data["y"] if "y" in data else None
    return X, y


def save_npz(path: str, X: np.ndarray, y: np.ndarray | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if y is None:
        np.savez(path, X=X)
    else:
        np.savez(path, X=X, y=y.astype(int))


def binarize_y(y):
    if y is None:
        return None
    y = np.asarray(y).astype(int)
    return (y > 0).astype(int)


def warn_or_raise(msg: str, strict: bool):
    if strict:
        raise ValueError(msg)
    print(f"[WARN] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine_dir", required=True,
                    help="VD: data/smd/machine-1-1 (chứa source.npz + target.npz)")
    ap.add_argument("--source_name", default="source.npz")
    ap.add_argument("--target_name", default="target.npz")

    ap.add_argument("--out_train", default="train_normal.npz")
    ap.add_argument("--out_val", default="val_mixed.npz")
    ap.add_argument("--out_test", default="test_mixed.npz")

    ap.add_argument("--train_normal_frac", type=float, default=1.0,
                    help="Tỉ lệ normal của SOURCE dùng làm train_normal (contiguous từ đầu).")
    ap.add_argument("--val_frac", type=float, default=0.3,
                    help="Tỉ lệ TARGET dùng làm val_mixed (contiguous từ đầu của target).")
    ap.add_argument("--guard", type=int, default=0,
                    help="Số mẫu bỏ qua giữa val và test trong TARGET (chống leakage).")

    # soft checks (default low for SMD machines)
    ap.add_argument("--min_train", type=int, default=0,
                    help="Nếu >0 thì check tối thiểu train_normal. (0 = không check)")
    ap.add_argument("--min_val", type=int, default=0)
    ap.add_argument("--min_test", type=int, default=0)

    ap.add_argument("--min_anom_val", type=int, default=1,
                    help="Tối thiểu số anomaly trong val (sau binarize).")
    ap.add_argument("--min_anom_test", type=int, default=1,
                    help="Tối thiểu số anomaly trong test (sau binarize).")

    ap.add_argument("--strict", action="store_true",
                    help="Bật để biến warning thành error (crash).")

    args = ap.parse_args()

    machine_dir = args.machine_dir
    src_path = os.path.join(machine_dir, args.source_name)
    tgt_path = os.path.join(machine_dir, args.target_name)

    Xs, ys = load_npz(src_path)
    Xt, yt = load_npz(tgt_path)

    if ys is None:
        raise ValueError(f"{src_path} must contain y to filter normal for train_normal.")
    if yt is None:
        raise ValueError(f"{tgt_path} must contain y to create mixed val/test.")

    ys = binarize_y(ys)
    yt = binarize_y(yt)

    # ---- train_normal from SOURCE: y==0 only, keep order
    idx_norm = np.where(ys == 0)[0]
    if len(idx_norm) == 0:
        raise ValueError("No normal samples (y==0) in source.")

    Xs_norm = Xs[idx_norm]
    frac = float(args.train_normal_frac)
    frac = max(0.0, min(1.0, frac))
    n_train = int(np.floor(len(Xs_norm) * frac))
    n_train = max(1, n_train)

    X_train = Xs_norm[:n_train]
    y_train = np.zeros(len(X_train), dtype=int)

    # ---- val/test from TARGET: contiguous split
    n_t = len(Xt)
    val_frac = float(args.val_frac)
    val_frac = max(0.0, min(0.95, val_frac))
    n_val = int(np.floor(n_t * val_frac))
    n_val = max(1, n_val)

    val_start = 0
    val_end = min(n_t, n_val)

    guard = max(0, int(args.guard))
    test_start = min(n_t, val_end + guard)
    test_end = n_t

    X_val = Xt[val_start:val_end]
    y_val = yt[val_start:val_end]

    X_test = Xt[test_start:test_end]
    y_test = yt[test_start:test_end]

    # ---- checks (soft by default)
    if args.min_train and len(X_train) < args.min_train:
        warn_or_raise(f"train_normal small: {len(X_train)} < min_train={args.min_train}", args.strict)
    if args.min_val and len(X_val) < args.min_val:
        warn_or_raise(f"val_mixed small: {len(X_val)} < min_val={args.min_val}", args.strict)
    if args.min_test and len(X_test) < args.min_test:
        warn_or_raise(f"test_mixed small: {len(X_test)} < min_test={args.min_test}", args.strict)

    anom_val = int((y_val == 1).sum())
    anom_test = int((y_test == 1).sum())
    if anom_val < args.min_anom_val:
        warn_or_raise(f"val_mixed few anomalies: {anom_val} < min_anom_val={args.min_anom_val}", args.strict)
    if anom_test < args.min_anom_test:
        warn_or_raise(f"test_mixed few anomalies: {anom_test} < min_anom_test={args.min_anom_test}", args.strict)

    out_train = os.path.join(machine_dir, args.out_train)
    out_val = os.path.join(machine_dir, args.out_val)
    out_test = os.path.join(machine_dir, args.out_test)

    save_npz(out_train, X_train, y_train)
    save_npz(out_val, X_val, y_val)
    save_npz(out_test, X_test, y_test)

    print("[DONE] Created UAD files (contiguous timeline):")
    print(f"  {out_train}: X={X_train.shape} y=(all 0) count={len(y_train)}")
    print(f"  {out_val}:   X={X_val.shape} y: normal={(y_val==0).sum()} anom={(y_val==1).sum()} count={len(y_val)}")
    print(f"  {out_test}:  X={X_test.shape} y: normal={(y_test==0).sum()} anom={(y_test==1).sum()} count={len(y_test)}")
    if guard > 0:
        print(f"  guard gap between val/test in target: {guard} samples")


if __name__ == "__main__":
    main()
