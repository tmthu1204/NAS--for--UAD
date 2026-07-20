import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from src.utils.data_paths import resolve_raw_hai_root


def _version_tag(version: str) -> str:
    return f"hai{str(version).replace('.', '_')}"


def _entity_name(version: str, stem: str) -> str:
    return f"{_version_tag(version)}-{stem}"


def _split_role(stem: str) -> str:
    stem = stem.lower()
    if stem.startswith("train"):
        return "train"
    if stem.startswith("test"):
        return "test"
    raise ValueError(f"Cannot infer HAI split role from file stem: {stem}")


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


def _csv_files(version_root: Path):
    files = sorted(list(version_root.glob("train*.csv")) + list(version_root.glob("train*.csv.gz")))
    files += sorted(list(version_root.glob("test*.csv")) + list(version_root.glob("test*.csv.gz")))
    dedup = {}
    for path in files:
        dedup[path.name] = path
    return [dedup[name] for name in sorted(dedup.keys())]


def _check_not_lfs_pointer(path: Path):
    opener = open
    kwargs = {"mode": "rt", "encoding": "utf-8", "errors": "replace"}
    if path.suffix.lower() == ".gz":
        import gzip

        opener = gzip.open
    with opener(path, **kwargs) as f:
        first = f.readline().strip()
    if first.startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"{path} is a Git LFS pointer, not real CSV content. "
            "Use a version with downloaded data (e.g. HAI 21.03 in this repo) or fetch LFS objects first."
        )


def _header_columns(path: Path):
    _check_not_lfs_pointer(path)
    return list(pd.read_csv(path, nrows=0).columns)


def _feature_columns(columns, attack_col: str):
    cols = []
    for col in columns:
        low = str(col).strip().lower()
        if low == "time":
            continue
        if low.startswith("attack"):
            continue
        cols.append(col)
    if attack_col not in columns:
        raise ValueError(f"Missing attack column '{attack_col}' in HAI CSV header.")
    if not cols:
        raise ValueError("No feature columns left after excluding time/attack columns.")
    return cols


def _accumulate_train_stats(train_files, feature_cols):
    total_rows = 0
    feat_sum = None
    feat_sumsq = None
    for path in train_files:
        X = pd.read_csv(path, usecols=feature_cols).to_numpy(dtype=np.float64)
        if feat_sum is None:
            feat_sum = np.zeros((X.shape[1],), dtype=np.float64)
            feat_sumsq = np.zeros((X.shape[1],), dtype=np.float64)
        feat_sum += X.sum(axis=0)
        feat_sumsq += np.square(X).sum(axis=0)
        total_rows += int(len(X))
        print(f"[STATS] {path.name}: rows={len(X)}")

    if total_rows == 0 or feat_sum is None or feat_sumsq is None:
        raise ValueError("No HAI train rows found for global normalization.")

    mean = feat_sum / float(total_rows)
    var = feat_sumsq / float(total_rows) - np.square(mean)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32), total_rows


def process_one_file(
    *,
    csv_path: Path,
    version: str,
    out_root: Path,
    feature_cols,
    attack_col: str,
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
    stride: int,
):
    stem = csv_path.name
    if stem.endswith(".csv.gz"):
        stem = stem[: -len(".csv.gz")]
    elif stem.endswith(".csv"):
        stem = stem[: -len(".csv")]

    role = _split_role(stem)
    df = pd.read_csv(csv_path, usecols=feature_cols + [attack_col])
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_rows = (pd.to_numeric(df[attack_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0).astype(np.int64)

    Xn = ((X - mean[None, :]) / std[None, :]).astype(np.float32)
    X_target = _to_windows(Xn, window, stride)
    y_target = _labels_to_windows(y_rows, window, stride, len(Xn))
    X_source = X_target[y_target == 0]
    y_source = np.zeros((len(X_source),), dtype=np.int64)

    if len(X_source) == 0:
        raise ValueError(f"{csv_path.name} has no fully-normal windows after window labeling.")

    entity_name = _entity_name(version, stem)
    entity_dir = out_root / entity_name
    entity_dir.mkdir(parents=True, exist_ok=True)

    np.savez(entity_dir / "source.npz", X=X_source, y=y_source)
    np.savez(entity_dir / "target.npz", X=X_target, y=y_target)

    metadata = {
        "dataset": "hai",
        "version": str(version),
        "version_tag": _version_tag(version),
        "entity_name": entity_name,
        "split_role": role,
        "raw_file": csv_path.name,
        "feature_columns": list(feature_cols),
        "feature_count": int(len(feature_cols)),
        "row_count": int(len(X)),
        "source_window_count": int(len(X_source)),
        "target_window_count": int(len(X_target)),
        "target_anomaly_window_count": int((y_target == 1).sum()),
        "window": int(window),
        "stride": int(stride),
        "attack_column": attack_col,
    }
    with open(entity_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[OK] {entity_name}: role={role} "
        f"source={X_source.shape} target={X_target.shape} positives={int(y_target.sum())}"
    )


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, default="external/hai")
    ap.add_argument("--version", type=str, default="21.03")
    ap.add_argument("--out_root", type=str, default="data/hai")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--attack_col", type=str, default="attack")
    return ap


def main():
    args = build_arg_parser().parse_args()

    version_root = resolve_raw_hai_root(args.raw_root, version=args.version)
    out_root = Path(args.out_root)
    csv_files = _csv_files(version_root)
    if not csv_files:
        raise FileNotFoundError(f"No HAI CSV files found under {version_root}")

    train_files = [p for p in csv_files if p.name.lower().startswith("train")]
    test_files = [p for p in csv_files if p.name.lower().startswith("test")]
    if not train_files or not test_files:
        raise ValueError(f"Expected both train*.csv(.gz) and test*.csv(.gz) under {version_root}")

    columns = _header_columns(train_files[0])
    feature_cols = _feature_columns(columns, args.attack_col)
    mean, std, total_train_rows = _accumulate_train_stats(train_files, feature_cols)

    out_root.mkdir(parents=True, exist_ok=True)
    norm_meta = {
        "dataset": "hai",
        "version": str(args.version),
        "version_root": str(version_root),
        "feature_count": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
        "attack_column": args.attack_col,
        "window": int(args.window),
        "stride": int(args.stride),
        "total_train_rows": int(total_train_rows),
    }
    with open(out_root / "normalization_metadata.json", "w", encoding="utf-8") as f:
        json.dump(norm_meta, f, indent=2)

    for csv_path in train_files + test_files:
        process_one_file(
            csv_path=csv_path,
            version=args.version,
            out_root=out_root,
            feature_cols=feature_cols,
            attack_col=args.attack_col,
            mean=mean,
            std=std,
            window=args.window,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
