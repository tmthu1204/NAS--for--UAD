from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def _read_txt_matrix(path: Path) -> np.ndarray:
    try:
        df = pd.read_csv(path, header=None, sep=r"[,\s]+", engine="python")
        arr = df.values.astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    except Exception as exc:
        raise FileNotFoundError(f"Cannot read data file: {path} ({exc})")


def _read_txt_labels(path: Path, expected_length: int) -> np.ndarray:
    try:
        df = pd.read_csv(path, header=None, sep=r"[,\s]+", engine="python")
        lab = df.values.reshape(-1).astype(np.int64)
        lab = (lab > 0).astype(np.int64)
        if lab.shape[0] != expected_length:
            raise ValueError(f"Label length mismatch: {path} has {lab.shape[0]} vs expected {expected_length}")
        return lab
    except Exception as exc:
        raise FileNotFoundError(f"Cannot read label file: {path} ({exc})")


def fit_train_zscore_apply(train: np.ndarray, test: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + eps
    return (train - mu) / sd, (test - mu) / sd


def fit_official_minmax_separately(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match the official OmniAnomaly repo preprocessing:
    train and test are each scaled independently with MinMaxScaler().
    """
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if np.any(np.isnan(train)):
        train = np.nan_to_num(train)
    if np.any(np.isnan(test)):
        test = np.nan_to_num(test)
    train_n = MinMaxScaler().fit_transform(train).astype(np.float32)
    test_n = MinMaxScaler().fit_transform(test).astype(np.float32)
    return train_n, test_n


def load_raw_smd_machine(raw_root: str | Path, machine: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_root = Path(raw_root)
    p_train = raw_root / "train" / f"{machine}.txt"
    p_test = raw_root / "test" / f"{machine}.txt"
    p_label = raw_root / "test_label" / f"{machine}.txt"

    x_train = _read_txt_matrix(p_train)
    x_test = _read_txt_matrix(p_test)
    y_test = _read_txt_labels(p_label, expected_length=x_test.shape[0])
    return x_train, x_test, y_test


def normalize_raw_smd_machine(
    raw_root: str | Path,
    machine: str,
    *,
    preprocess_mode: str = "official_minmax",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_test = load_raw_smd_machine(raw_root, machine)
    if preprocess_mode == "official_minmax":
        x_train_n, x_test_n = fit_official_minmax_separately(x_train, x_test)
    elif preprocess_mode == "train_zscore":
        x_train_n, x_test_n = fit_train_zscore_apply(x_train, x_test)
    else:
        raise ValueError(f"Unknown preprocess_mode={preprocess_mode}")
    return x_train_n.astype(np.float32), x_test_n.astype(np.float32), y_test.astype(np.int64)


def contiguous_train_valid_split(x_train: np.ndarray, valid_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    n = len(x_train)
    n_val = max(1, int(round(n * valid_ratio)))
    n_train = max(1, n - n_val)
    return x_train[:n_train], x_train[n_train:]


def aligned_last_point_labels(y_test: np.ndarray, window_length: int, stride: int = 1) -> np.ndarray:
    y_test = np.asarray(y_test).astype(np.int64)
    if len(y_test) < window_length:
        return np.asarray([int(y_test.max() > 0)], dtype=np.int64)

    starts = np.arange(0, len(y_test) - window_length + 1, stride, dtype=int)
    return y_test[starts + window_length - 1]


class SlidingWindowDataset(Dataset):
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

    def __len__(self):
        return int(len(self.starts))

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + self.window_length
        x = self.series[start:end]
        if len(x) < self.window_length:
            pad = np.zeros((self.window_length - len(x), self.series.shape[1]), dtype=self.series.dtype)
            x = np.concatenate([x, pad], axis=0)
        return torch.from_numpy(x.astype(np.float32))


@dataclass
class RawSMDMachine:
    machine: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @classmethod
    def from_root(
        cls,
        raw_root: str | Path,
        machine: str,
        *,
        preprocess_mode: str = "official_minmax",
    ) -> "RawSMDMachine":
        x_train, x_test, y_test = normalize_raw_smd_machine(
            raw_root,
            machine,
            preprocess_mode=preprocess_mode,
        )
        return cls(machine=machine, x_train=x_train, x_test=x_test, y_test=y_test)
