import argparse
import ast
from collections import defaultdict
from pathlib import Path
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from src.utils.data_paths import resolve_raw_smap_root


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
    return np.stack([X[s : s + win] for s in starts], axis=0)


def _labels_to_windows(y: np.ndarray, win: int, stride: int, T: int) -> np.ndarray:
    if T < win:
        return np.array([int(y.sum() > 0)], dtype=np.int64)
    starts = np.arange(0, T - win + 1, stride, dtype=int)
    return np.array([int(y[s : s + win].max() > 0) for s in starts], dtype=np.int64)


def load_telemanom_channel_metadata(raw_root: Path, spacecraft: str):
    import csv

    rows_by_channel: dict[str, list[dict]] = defaultdict(list)
    csv_path = raw_root / "labeled_anomalies.csv"
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("spacecraft") != spacecraft:
                continue
            chan_id = row["chan_id"].strip()
            rows_by_channel[chan_id].append(row)
    return rows_by_channel


def build_test_labels(rows: list[dict], test_len: int) -> np.ndarray:
    y = np.zeros(test_len, dtype=np.int64)
    for row in rows:
        try:
            sequences = ast.literal_eval(row["anomaly_sequences"])
        except Exception as exc:
            raise ValueError(f"Cannot parse anomaly_sequences for {row.get('chan_id')}: {exc}")
        for start, end in sequences:
            start = max(0, int(start))
            end = min(test_len - 1, int(end))
            if end >= start:
                y[start : end + 1] = 1
    return y


def available_spacecraft_channels(raw_root: Path, rows_by_channel: dict[str, list[dict]]):
    train_dir = raw_root / "train"
    test_dir = raw_root / "test"
    chans = []
    for chan_id in sorted(rows_by_channel.keys()):
        if (train_dir / f"{chan_id}.npy").exists() and (test_dir / f"{chan_id}.npy").exists():
            chans.append(chan_id)
    return chans


def process_one_channel(raw_root: Path, chan_id: str, out_root: Path, rows_by_channel, *, win: int, stride: int):
    train_path = raw_root / "train" / f"{chan_id}.npy"
    test_path = raw_root / "test" / f"{chan_id}.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test file for channel {chan_id}")

    Xtr = np.load(train_path).astype(np.float32)
    Xte = np.load(test_path).astype(np.float32)
    if Xtr.ndim == 1:
        Xtr = Xtr[:, None]
    if Xte.ndim == 1:
        Xte = Xte[:, None]

    rows = rows_by_channel.get(chan_id)
    if not rows:
        raise KeyError(f"No labeled metadata row found for SMAP channel {chan_id}")
    yte = build_test_labels(rows, test_len=Xte.shape[0])

    Xtr_n, Xte_n = _zscore_fit_apply(Xtr, Xte)
    Xs = _to_windows(Xtr_n, win, stride)
    Xt = _to_windows(Xte_n, win, stride)
    yt = _labels_to_windows(yte, win, stride, Xte.shape[0])
    ys = np.zeros((Xs.shape[0],), dtype=np.int64)

    out_dir = out_root / chan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "source.npz", X=Xs, y=ys)
    np.savez(out_dir / "target.npz", X=Xt, y=yt)
    print(f"[OK] {chan_id}: source {Xs.shape}, target {Xt.shape}, positives={yt.sum()}")


def run_preprocess(*, raw_root: Path, out_root: Path, spacecraft: str, channel: str | None, window: int, stride: int):
    rows_by_channel = load_telemanom_channel_metadata(raw_root, spacecraft)
    channels = [channel] if channel else available_spacecraft_channels(raw_root, rows_by_channel)
    if not channels:
        raise FileNotFoundError(f"No {spacecraft} channels with train/test data found under {raw_root}")

    print(
        f"[INFO] {spacecraft} raw layout detected. Channels={len(channels)} | "
        f"window={window} stride={stride}"
    )
    for chan_id in channels:
        process_one_channel(
            raw_root,
            chan_id,
            out_root,
            rows_by_channel,
            win=window,
            stride=stride,
        )


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_root",
        type=str,
        default="data/SMAP_MSL",
        help="Directory containing train/, test/, and labeled_anomalies.csv for Telemanom SMAP/MSL.",
    )
    ap.add_argument("--out_root", type=str, default=None)
    ap.add_argument("--dataset", choices=["SMAP", "MSL"], default="SMAP")
    ap.add_argument("--channel", type=str, default=None, help="Example: A-1. If omitted, preprocess all SMAP channels.")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    return ap


def main():
    args = build_arg_parser().parse_args()

    raw_root = resolve_raw_smap_root(args.raw_root)
    out_root = Path(args.out_root) if args.out_root else Path("data") / args.dataset.lower()

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw Telemanom root not found: {raw_root}")

    run_preprocess(
        raw_root=raw_root,
        out_root=out_root,
        spacecraft=args.dataset,
        channel=args.channel,
        window=args.window,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
