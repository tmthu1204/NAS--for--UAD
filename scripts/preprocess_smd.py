# scripts/preprocess_smd.py
import argparse
from pathlib import Path
import numpy as np

def _read_txt_matrix(p: Path) -> np.ndarray:
    import pandas as pd
    try:
        # tách bằng comma HOẶC whitespace; engine='python' chịu được dòng có trailing comma
        df = pd.read_csv(p, header=None, sep=r"[,\s]+", engine="python")
        arr = df.values.astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    except Exception as e:
        raise FileNotFoundError(f"Cannot read data file: {p} ({e})")

def _read_txt_labels(p: Path, T_expected: int) -> np.ndarray:
    import pandas as pd
    try:
        df = pd.read_csv(p, header=None, sep=r"[,\s]+", engine="python")
        lab = df.values.reshape(-1).astype(np.int64)   # flatten nếu nhiều cột
        lab = (lab > 0).astype(np.int64)
        if lab.shape[0] != T_expected:
            raise ValueError(f"Label length mismatch: {p} has {lab.shape[0]} vs test T={T_expected}")
        return lab
    except Exception as e:
        raise FileNotFoundError(f"Cannot read label file: {p} ({e})")


def _read_csv_matrix(p: Path) -> np.ndarray:
    import pandas as pd
    try:
        arr = pd.read_csv(p, header=None).values.astype(np.float32)
        if arr.ndim == 1: arr = arr[:, None]
        return arr
    except Exception as e:
        raise FileNotFoundError(f"Cannot read csv file: {p} ({e})")

def _zscore_fit_apply(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd

def _to_windows(X: np.ndarray, win: int, stride: int) -> np.ndarray:
    T, C = X.shape
    if T < win:
        pad = np.zeros((win - T, C), dtype=X.dtype)
        Xp = np.concatenate([X, pad], axis=0)
        return Xp[None, ...]
    starts = np.arange(0, T - win + 1, stride, dtype=int)
    return np.stack([X[s:s+win] for s in starts], axis=0)

def _labels_to_windows(y: np.ndarray, win: int, stride: int, T: int) -> np.ndarray:
    if T < win:
        return np.array([int(y.sum() > 0)], dtype=np.int64)
    starts = np.arange(0, T - win + 1, stride, dtype=int)
    return np.array([int(y[s:s+win].max() > 0) for s in starts], dtype=np.int64)

def process_one_omni(root: Path, machine: str, out_root: Path, win=128, stride=64):
    """OmniAnomaly layout: root/train|test|test_label/machine-*.txt"""
    p_train = root / "train" / f"{machine}.txt"
    p_test  = root / "test" / f"{machine}.txt"
    p_lab   = root / "test_label" / f"{machine}.txt"

    Xtr = _read_txt_matrix(p_train)
    Xte = _read_txt_matrix(p_test)
    yte = _read_txt_labels(p_lab, T_expected=Xte.shape[0])

    Xtr_n, Xte_n = _zscore_fit_apply(Xtr, Xte)
    Xs = _to_windows(Xtr_n, win, stride)
    Xt = _to_windows(Xte_n, win, stride)
    yt = _labels_to_windows(yte, win, stride, Xte.shape[0])
    ys = np.zeros((Xs.shape[0],), dtype=np.int64)

    out_dir = out_root / machine
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "source.npz", X=Xs, y=ys)
    np.savez(out_dir / "target.npz", X=Xt, y=yt)
    print(f"[OK] {machine}: source {Xs.shape}, target {Xt.shape}, positives={yt.sum()}")

def process_one_csv(machine_dir: Path, out_dir: Path, win=128, stride=64):
    """Legacy CSV layout: machine-*/train.csv, test.csv, (test_label|labels|label).csv"""
    Xtr = _read_csv_matrix(machine_dir / "train.csv")
    Xte = _read_csv_matrix(machine_dir / "test.csv")

    # labels variants
    lab = None
    for name in ["test_label.csv", "labels.csv", "label.csv"]:
        p = machine_dir / name
        if p.exists():
            import pandas as pd
            lab = pd.read_csv(p, header=None).values.reshape(-1)
            break
    if lab is None:
        raise FileNotFoundError(f"No label csv found in {machine_dir}")

    lab = (lab > 0).astype(np.int64)
    if lab.shape[0] != Xte.shape[0]:
        raise ValueError(f"CSV label length mismatch in {machine_dir}")

    Xtr_n, Xte_n = _zscore_fit_apply(Xtr, Xte)
    Xs = _to_windows(Xtr_n, win, stride)
    Xt = _to_windows(Xte_n, win, stride)
    yt = _labels_to_windows(lab, win, stride, Xte.shape[0])
    ys = np.zeros((Xs.shape[0],), dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "source.npz", X=Xs, y=ys)
    np.savez(out_dir / "target.npz", X=Xt, y=yt)
    print(f"[OK] {machine_dir.name}: source {Xs.shape}, target {Xt.shape}, positives={yt.sum()}")

def list_omni_machines(root: Path):
    # dựa theo danh sách file trong train/
    train_dir = root / "train"
    return sorted([p.stem for p in train_dir.glob("*.txt")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, default="data/ServerMachineDataset",
                    help="Thư mục chứa train/test/test_label (OmniAnomaly). "
                         "Hoặc thư mục chứa các machine-*/train.csv (legacy).")
    ap.add_argument("--out_root", type=str, default="data/smd")
    ap.add_argument("--machine", type=str, default=None, help="VD: machine-1-1; nếu bỏ trống sẽ xử lý tất cả.")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    # Phát hiện layout OmniAnomaly vs CSV legacy
    is_omni = (raw_root / "train").exists() and (raw_root / "test").exists() and (raw_root / "test_label").exists()

    if is_omni:
        machines = [args.machine] if args.machine else list_omni_machines(raw_root)
        if not machines:
            raise FileNotFoundError(f"No *.txt machines in {raw_root/'train'}")
        print(f"[INFO] OmniAnomaly layout detected. Machines={len(machines)} | window={args.window} stride={args.stride}")
        for m in machines:
            process_one_omni(raw_root, m, out_root, win=args.window, stride=args.stride)
    else:
        # legacy: raw_root/machine-1-1/{train.csv,test.csv,labels.csv}
        if args.machine:
            mdir = raw_root / args.machine
            if not mdir.exists():
                raise FileNotFoundError(f"Machine folder not found: {mdir}")
            process_one_csv(mdir, out_root / mdir.name, win=args.window, stride=args.stride)
        else:
            mdirs = [p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("machine-")]
            if not mdirs:
                raise FileNotFoundError(f"No machine-* folders in {raw_root}")
            print(f"[INFO] CSV legacy layout detected. Machines={len(mdirs)}")
            for mdir in mdirs:
                process_one_csv(mdir, out_root / mdir.name, win=args.window, stride=args.stride)

if __name__ == "__main__":
    main()
