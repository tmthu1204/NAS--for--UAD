from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _read_txt_matrix(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError:
        arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.asarray(arr, dtype=np.float64)


def _read_txt_labels(path: Path, expected_length: int) -> np.ndarray:
    try:
        lab = np.loadtxt(path, delimiter=",", dtype=np.int64)
    except ValueError:
        lab = np.loadtxt(path, dtype=np.int64)
    lab = np.asarray(lab).reshape(-1)
    lab = (lab > 0).astype(np.int64)
    if lab.shape[0] != expected_length:
        raise ValueError(f"Label length mismatch: {path} has {lab.shape[0]} vs expected {expected_length}")
    return lab


def load_raw_tranad_smd_machine(raw_root: str | Path, machine: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_root = Path(raw_root)
    x_train = _read_txt_matrix(raw_root / "train" / f"{machine}.txt")
    x_test = _read_txt_matrix(raw_root / "test" / f"{machine}.txt")
    y_test = _read_txt_labels(raw_root / "test_label" / f"{machine}.txt", expected_length=x_test.shape[0])
    return x_train, x_test, y_test


def build_tranad_windows(series: np.ndarray, window_length: int) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(f"Expected 2-D series [T, C], got shape {series.shape}")
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")

    n_steps, n_feats = series.shape
    windows = np.empty((n_steps, window_length, n_feats), dtype=np.float64)
    first_row = series[0:1]
    for i in range(n_steps):
        if i >= window_length:
            windows[i] = series[i - window_length:i]
        else:
            pad = np.repeat(first_row, window_length - i, axis=0)
            hist = series[0:i]
            windows[i] = np.concatenate([pad, hist], axis=0)
    return windows
