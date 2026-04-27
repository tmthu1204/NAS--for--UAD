from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def _read_csv_with_fallback(path: Path, *, preferred_sep: str | None = None) -> pd.DataFrame:
    candidates = []
    if preferred_sep is not None:
        candidates.append(preferred_sep)
    candidates.extend([None, ";", ","])

    seen = set()
    for sep in candidates:
        key = "__auto__" if sep is None else sep
        if key in seen:
            continue
        seen.add(key)
        try:
            kwargs = {"low_memory": False}
            if sep is not None:
                kwargs["sep"] = sep
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise FileNotFoundError(f"Cannot read SWaT CSV: {path}")


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(series, errors="coerce")
    if out.isna().values.any():
        out = out.ffill().bfill().fillna(0.0)
    return out.astype(np.float32)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["Timestamp", "Normal/Attack"] if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def _downsample_median(x: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if factor <= 1 or len(x) == 0:
        return x.astype(np.float32)

    chunks = []
    for start in range(0, len(x), factor):
        chunk = x[start:start + factor]
        chunks.append(np.median(chunk, axis=0))
    return np.stack(chunks, axis=0).astype(np.float32)


def _downsample_labels_any(y: np.ndarray, factor: int) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    if factor <= 1 or len(y) == 0:
        return y

    chunks = []
    for start in range(0, len(y), factor):
        chunk = y[start:start + factor]
        chunks.append(int(np.any(chunk > 0)))
    return np.asarray(chunks, dtype=np.int64)


def read_swat_normal_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    df = _normalize_column_names(_read_csv_with_fallback(path, preferred_sep=None))
    feats = _coerce_numeric_frame(_feature_frame(df))
    return feats.to_numpy(dtype=np.float32, copy=True)


def read_swat_attack_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    df = _normalize_column_names(_read_csv_with_fallback(path, preferred_sep=";"))
    if "Normal/Attack" not in df.columns:
        raise ValueError(f"SWaT attack CSV must contain 'Normal/Attack': {path}")

    labels = (
        df["Normal/Attack"]
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("normal")
        .astype(np.int64)
        .to_numpy()
    )
    feats = _coerce_numeric_frame(_feature_frame(df))
    return feats.to_numpy(dtype=np.float32, copy=True), labels


def fit_train_minmax_apply(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    train_n = scaler.fit_transform(np.asarray(train, dtype=np.float32)).astype(np.float32)
    test_n = scaler.transform(np.asarray(test, dtype=np.float32)).astype(np.float32)
    return train_n, test_n


def fit_train_zscore_apply(train: np.ndarray, test: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + eps
    return ((train - mu) / sd).astype(np.float32), ((test - mu) / sd).astype(np.float32)


def load_raw_swat(
    train_csv: str | Path,
    test_csv: str | Path,
    *,
    preprocess_mode: str = "train_minmax",
    downsample: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = read_swat_normal_csv(train_csv)
    x_test, y_test = read_swat_attack_csv(test_csv)

    if preprocess_mode == "train_minmax":
        x_train, x_test = fit_train_minmax_apply(x_train, x_test)
    elif preprocess_mode == "train_zscore":
        x_train, x_test = fit_train_zscore_apply(x_train, x_test)
    else:
        raise ValueError(f"Unknown preprocess_mode={preprocess_mode}")

    x_train = _downsample_median(x_train, max(1, int(downsample)))
    x_test = _downsample_median(x_test, max(1, int(downsample)))
    y_test = _downsample_labels_any(y_test, max(1, int(downsample)))

    if len(x_test) != len(y_test):
        raise ValueError(
            f"SWaT test/label length mismatch after downsampling: {len(x_test)} vs {len(y_test)}"
        )

    return x_train.astype(np.float32), x_test.astype(np.float32), y_test.astype(np.int64)


def upstream_usad_window_starts(length: int, window_length: int, stride: int = 1) -> np.ndarray:
    """
    Match the upstream USAD SWaT notebook window construction:
      np.arange(window_size)[None, :] + np.arange(T - window_size)[:, None]

    This yields starts in [0, T - window_length), i.e. the last full-length window
    is intentionally omitted. For very short inputs we keep a single fallback window
    to avoid empty tensors in quick tests.
    """
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if length <= 0:
        return np.empty((0,), dtype=np.int64)
    if length <= window_length:
        return np.asarray([0], dtype=np.int64)
    return np.arange(0, length - window_length, stride, dtype=np.int64)


def build_upstream_usad_flat_windows(
    series: np.ndarray,
    window_length: int,
    stride: int = 1,
) -> np.ndarray:
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 2:
        raise ValueError(f"Expected 2-D series [T, C], got shape {series.shape}")

    starts = upstream_usad_window_starts(len(series), window_length, stride)
    if starts.size == 0:
        return np.empty((0, window_length * series.shape[1]), dtype=np.float32)

    if len(series) <= window_length:
        x = series
        if len(x) < window_length:
            pad = np.zeros((window_length - len(x), series.shape[1]), dtype=series.dtype)
            x = np.concatenate([x, pad], axis=0)
        return x.reshape(1, -1).astype(np.float32)

    idx = starts[:, None] + np.arange(window_length, dtype=np.int64)[None, :]
    windows = series[idx]
    return windows.reshape(windows.shape[0], -1).astype(np.float32, copy=False)


def build_upstream_usad_window_labels(
    labels: np.ndarray,
    window_length: int,
    stride: int = 1,
) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64)
    starts = upstream_usad_window_starts(len(labels), window_length, stride)
    if starts.size == 0:
        return np.empty((0,), dtype=np.int64)

    if len(labels) <= window_length:
        return np.asarray([int(np.any(labels > 0))], dtype=np.int64)

    idx = starts[:, None] + np.arange(window_length, dtype=np.int64)[None, :]
    return (labels[idx].sum(axis=1) > 0).astype(np.int64)


class FlattenedSlidingWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, window_length: int, stride: int = 1):
        series = np.asarray(series, dtype=np.float32)
        if series.ndim != 2:
            raise ValueError(f"Expected 2-D series [T, C], got shape {series.shape}")
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.series = series
        self.window_length = int(window_length)
        self.stride = int(stride)

        if len(series) < window_length:
            self.starts = np.asarray([0], dtype=np.int64)
        else:
            self.starts = np.arange(0, len(series) - window_length + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        end = start + self.window_length
        x = self.series[start:end]
        if len(x) < self.window_length:
            pad = np.zeros((self.window_length - len(x), self.series.shape[1]), dtype=self.series.dtype)
            x = np.concatenate([x, pad], axis=0)
        return torch.from_numpy(x.reshape(-1).astype(np.float32))


@dataclass
class RawSWaTDataset:
    train_csv: str
    test_csv: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @classmethod
    def from_csvs(
        cls,
        train_csv: str | Path,
        test_csv: str | Path,
        *,
        preprocess_mode: str = "train_minmax",
        downsample: int = 5,
    ) -> "RawSWaTDataset":
        x_train, x_test, y_test = load_raw_swat(
            train_csv,
            test_csv,
            preprocess_mode=preprocess_mode,
            downsample=downsample,
        )
        return cls(
            train_csv=str(train_csv),
            test_csv=str(test_csv),
            x_train=x_train,
            x_test=x_test,
            y_test=y_test,
        )
