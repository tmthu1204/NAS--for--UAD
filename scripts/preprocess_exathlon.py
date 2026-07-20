import argparse
import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from src.utils.data_paths import resolve_raw_exathlon_root


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


def _fit_normalizer(X: np.ndarray, y: np.ndarray, mode: str):
    if mode == "normal_rows" and np.any(y == 0):
        ref = X[y == 0]
    else:
        ref = X
    mu = ref.mean(axis=0, keepdims=True)
    sd = ref.std(axis=0, keepdims=True) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def _parse_app_index(app_name: str) -> int:
    match = re.search(r"(\d+)$", app_name.lower())
    if not match:
        raise ValueError(f"Cannot parse app index from {app_name}")
    return int(match.group(1))


def _trace_entity_name(app_name: str, trace_stem: str) -> str:
    parts = trace_stem.split("_")
    if len(parts) >= 4:
        return f"{app_name}-{parts[1]}-{parts[2]}-{parts[3]}"
    return f"{app_name}-{trace_stem.replace('_', '-')}"


def _trace_meta(app_name: str, trace_stem: str) -> dict:
    parts = trace_stem.split("_")
    app_idx = _parse_app_index(app_name)
    meta = {
        "app_name": app_name,
        "app_index": app_idx,
        "raw_trace_name": trace_stem,
        "entity_name": _trace_entity_name(app_name, trace_stem),
    }
    if len(parts) >= 4:
        meta.update(
            {
                "trace_app_index": int(parts[0]) if parts[0].isdigit() else app_idx,
                "type_id": int(parts[1]) if parts[1].isdigit() else None,
                "input_rate": int(parts[2]) if parts[2].isdigit() else None,
                "trace_id": int(parts[3]) if parts[3].isdigit() else None,
            }
        )
    else:
        meta.update({"trace_app_index": app_idx, "type_id": None, "input_rate": None, "trace_id": None})
    return meta


def _load_ground_truth_rows(data_root: Path):
    rows_by_trace: dict[str, list[dict]] = {}
    gt_csv = data_root / "ground_truth.csv"
    gt_zip = data_root / "raw" / "ground_truth.zip"
    if gt_csv.exists():
        with open(gt_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace_name = str(row.get("trace_name", "")).strip()
                if not trace_name:
                    continue
                rows_by_trace.setdefault(trace_name, []).append(row)
        return rows_by_trace

    if gt_zip.exists():
        with zipfile.ZipFile(gt_zip, "r") as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
            if not members:
                raise ValueError(f"No CSV found inside {gt_zip}")
            with zf.open(members[0], "r") as f:
                reader = csv.DictReader((line.decode("utf-8") for line in f))
                for row in reader:
                    trace_name = str(row.get("trace_name", "")).strip()
                    if not trace_name:
                        continue
                    rows_by_trace.setdefault(trace_name, []).append(row)
        return rows_by_trace

    raise FileNotFoundError(f"Missing ground_truth.csv or raw/ground_truth.zip under {data_root}")


def _normalize_root(raw_root: Path) -> Path:
    if (raw_root / "ground_truth.csv").exists() or (raw_root / "raw" / "ground_truth.zip").exists():
        return raw_root
    data_dir = raw_root / "data"
    if ((data_dir / "ground_truth.csv").exists() or (data_dir / "raw" / "ground_truth.zip").exists()) and (data_dir / "raw").exists():
        return data_dir
    raise FileNotFoundError(
        f"Could not find Exathlon raw layout under {raw_root}. Expected either "
        f"'ground_truth.csv' next to app folders, or 'data/raw/ground_truth.zip' with 'data/raw/'."
    )


def _trace_files(app_dir: Path) -> list[Path]:
    files = {}
    for pattern in ("*.csv", "*.zip"):
        for path in sorted(app_dir.glob(pattern)):
            files.setdefault(path.stem, path)
    return [files[k] for k in sorted(files.keys())]


def _read_trace_frame(trace_path: Path) -> pd.DataFrame:
    if trace_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(trace_path, "r") as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
            if not members:
                raise ValueError(f"No CSV found inside {trace_path}")
            with zf.open(members[0], "r") as f:
                return pd.read_csv(f)
    return pd.read_csv(trace_path)


def _detect_timestamp_column(df: pd.DataFrame) -> str:
    exact = {
        "timestamp",
        "time",
        "t",
        "datetime",
        "date",
        "record_timestamp",
        "event_time",
    }
    lower_to_col = {str(c).strip().lower(): c for c in df.columns}
    for name in exact:
        if name in lower_to_col:
            return lower_to_col[name]

    for col in df.columns:
        cl = str(col).strip().lower()
        if "time" in cl or "timestamp" in cl:
            return col

    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            continue
        vals = numeric.to_numpy(dtype=np.float64)
        if len(vals) == 0:
            continue
        if np.all(np.diff(vals) >= 0) and np.nanmax(vals) >= 1e8:
            return col

    raise ValueError("Could not infer timestamp column from trace CSV")


def _to_unix_seconds(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float64)
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().all():
        return (dt.astype("int64") // 10**9).to_numpy(dtype=np.float64)
    raise ValueError("Timestamp column is neither numeric nor parseable datetime")


def _safe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except Exception:
        try:
            return float(pd.to_datetime(text, utc=True).value / 10**9)
        except Exception:
            return None


def _build_row_labels(timestamps: np.ndarray, gt_rows: list[dict]) -> np.ndarray:
    y = np.zeros((len(timestamps),), dtype=np.int64)
    if not gt_rows:
        return y
    for row in gt_rows:
        start = _safe_float(row.get("root_cause_start"))
        end = _safe_float(row.get("extended_effect_end"))
        if end is None:
            end = _safe_float(row.get("root_cause_end"))
        if start is None or end is None:
            continue
        lo = min(start, end)
        hi = max(start, end)
        y[(timestamps >= lo) & (timestamps <= hi)] = 1
    return y


def _feature_frame(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    feat_df = df.drop(columns=[timestamp_col]).copy()
    numeric_blocks = []
    kept_cols = []
    for col in feat_df.columns:
        block = pd.to_numeric(feat_df[col], errors="coerce")
        numeric_blocks.append(block)
        kept_cols.append(col)
    if not numeric_blocks:
        raise ValueError("No numeric feature columns found in trace CSV")
    out = pd.concat(numeric_blocks, axis=1)
    out.columns = kept_cols
    out = out.ffill().bfill().fillna(0.0)
    return out


def process_one_trace(
    *,
    data_root: Path,
    app_dir: Path,
    trace_path: Path,
    out_root: Path,
    gt_rows_by_trace: dict[str, list[dict]],
    window: int,
    stride: int,
    scale_on: str,
):
    app_name = app_dir.name
    trace_stem = trace_path.stem
    meta = _trace_meta(app_name, trace_stem)

    df = _read_trace_frame(trace_path)
    timestamp_col = _detect_timestamp_column(df)
    timestamps = _to_unix_seconds(df[timestamp_col])
    row_labels = _build_row_labels(timestamps, gt_rows_by_trace.get(trace_stem, []))
    feat_df = _feature_frame(df, timestamp_col)

    X = feat_df.to_numpy(dtype=np.float32)
    mu, sd = _fit_normalizer(X, row_labels, scale_on)
    Xn = ((X - mu) / sd).astype(np.float32)

    X_target = _to_windows(Xn, window, stride)
    y_target = _labels_to_windows(row_labels, window, stride, len(Xn))
    X_source = X_target[y_target == 0]
    y_source = np.zeros((len(X_source),), dtype=np.int64)

    if len(X_source) == 0:
        raise ValueError(f"{trace_path.name} has no fully-normal windows after labeling")

    entity_dir = out_root / meta["entity_name"]
    entity_dir.mkdir(parents=True, exist_ok=True)
    np.savez(entity_dir / "source.npz", X=X_source, y=y_source)
    np.savez(entity_dir / "target.npz", X=X_target, y=y_target)

    entity_meta = {
        **meta,
        "timestamp_column": str(timestamp_col),
        "row_count": int(len(Xn)),
        "feature_count": int(Xn.shape[1]),
        "source_window_count": int(len(X_source)),
        "target_window_count": int(len(X_target)),
        "target_anomaly_window_count": int((y_target == 1).sum()),
        "scale_on": scale_on,
        "window": int(window),
        "stride": int(stride),
        "raw_path": str(trace_path.relative_to(data_root.parent if data_root.name == "raw" else data_root)),
    }
    with open(entity_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(entity_meta, f, indent=2)

    print(
        f"[OK] {entity_meta['entity_name']}: "
        f"source={X_source.shape}, target={X_target.shape}, positives={int(y_target.sum())}"
    )


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_root",
        type=str,
        default="external/exathlon",
        help="Exathlon repo root, data root, or raw root. The script accepts either "
        "'.../exathlon', '.../exathlon/data', or a folder containing ground_truth.csv and app*/ traces.",
    )
    ap.add_argument("--out_root", type=str, default="data/exathlon")
    ap.add_argument("--app", type=str, default=None, help="Optional single app folder, e.g. app1")
    ap.add_argument("--trace", type=str, default=None, help="Optional raw trace stem filter, e.g. 1_0_1000000_14")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument(
        "--scale_on",
        choices=["normal_rows", "full_trace"],
        default="normal_rows",
        help="Fit z-score parameters on only normal rows when available, or on the full trace.",
    )
    return ap


def main():
    args = build_arg_parser().parse_args()

    raw_root = resolve_raw_exathlon_root(args.raw_root)
    data_root = _normalize_root(raw_root)
    raw_dir = data_root / "raw" if (data_root / "raw").exists() else data_root
    out_root = Path(args.out_root)

    gt_rows_by_trace = _load_ground_truth_rows(data_root)
    app_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir() and p.name.lower().startswith("app")])
    if args.app:
        app_dirs = [p for p in app_dirs if p.name == args.app]
    if not app_dirs:
        raise FileNotFoundError(f"No app* directories found under {raw_dir}")

    print(
        f"[INFO] Exathlon raw root detected. apps={len(app_dirs)} | "
        f"window={args.window} stride={args.stride}"
    )

    processed = 0
    for app_dir in app_dirs:
        for trace_path in _trace_files(app_dir):
            if args.trace and trace_path.stem != args.trace:
                continue
            process_one_trace(
                data_root=data_root,
                app_dir=app_dir,
                trace_path=trace_path,
                out_root=out_root,
                gt_rows_by_trace=gt_rows_by_trace,
                window=args.window,
                stride=args.stride,
                scale_on=args.scale_on,
            )
            processed += 1

    if processed == 0:
        raise ValueError("No Exathlon traces were processed. Check --app/--trace filters.")


if __name__ == "__main__":
    main()
