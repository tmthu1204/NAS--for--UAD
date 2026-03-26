from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import genpareto


@dataclass
class SPOTResult:
    threshold_dynamic: np.ndarray
    alarms: List[int]
    init_threshold: float


class SPOT:
    """
    Lightweight SPOT/POT implementation for Omni-style evaluation.

    This is not the original TensorFlow/tfsnippet implementation, but it follows
    the same high-level protocol:
    - fit on initialization scores
    - choose a high quantile initial threshold
    - fit a GPD on excesses
    - update the threshold online on non-alarm excesses
    """

    def __init__(self, q: float = 1e-3):
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        self.q = float(q)
        self.init_data = None
        self.data = None
        self._init_data_proc = None
        self._data_proc = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.nt = 0
        self.min_extrema = False

    def fit(self, init_data, data):
        self.init_data = np.asarray(init_data).astype(float).reshape(-1)
        self.data = np.asarray(data).astype(float).reshape(-1)
        if self.init_data.size == 0:
            raise ValueError("init_data must be non-empty")
        if self.data.size == 0:
            raise ValueError("data must be non-empty")

    def initialize(self, level: float = 0.98, min_extrema: bool = False):
        if self.init_data is None:
            raise RuntimeError("Call fit() before initialize().")
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")

        self.min_extrema = bool(min_extrema)
        if self.min_extrema:
            self._init_data_proc = -self.init_data
            self._data_proc = -self.data
        else:
            self._init_data_proc = self.init_data
            self._data_proc = self.data

        self.init_threshold = float(np.quantile(self._init_data_proc, level))
        excesses = self._init_data_proc[self._init_data_proc > self.init_threshold] - self.init_threshold
        if excesses.size == 0:
            excesses = np.asarray([1e-6], dtype=float)

        self.peaks = excesses.astype(float)
        self.n = int(self._init_data_proc.size)
        self.nt = int(self.peaks.size)
        return self

    def _fit_gpd(self):
        peaks = np.asarray(self.peaks).astype(float)
        peaks = peaks[np.isfinite(peaks)]
        if peaks.size == 0:
            return 1e-6, 0.0
        try:
            shape, _, scale = genpareto.fit(peaks, floc=0.0)
            scale = float(max(scale, 1e-8))
            shape = float(shape)
            return scale, shape
        except Exception:
            return float(np.std(peaks) + 1e-6), 0.0

    def _extreme_quantile(self):
        scale, shape = self._fit_gpd()
        if self.nt <= 0:
            return float(self.init_threshold)

        ratio = self.q * self.n / self.nt
        ratio = max(ratio, 1e-12)

        if abs(shape) < 1e-8:
            y = -scale * np.log(ratio)
        else:
            y = scale / shape * (ratio ** (-shape) - 1.0)
        return float(self.init_threshold + y)

    def run(self, with_alarm: bool = True, dynamic: bool = True) -> Dict[str, np.ndarray | List[int] | float]:
        if self._data_proc is None or self.init_threshold is None or self.peaks is None:
            raise RuntimeError("Call fit() and initialize() before run().")

        thresholds = []
        alarms: List[int] = []
        current_threshold = self._extreme_quantile()

        for i, x in enumerate(self._data_proc):
            x = float(x)
            thresholds.append(current_threshold)

            if x > current_threshold:
                if with_alarm:
                    alarms.append(i)
                continue

            if dynamic and x > self.init_threshold:
                self.peaks = np.append(self.peaks, x - self.init_threshold)
                self.n += 1
                self.nt += 1
                current_threshold = self._extreme_quantile()
            else:
                self.n += 1

        return {
            "thresholds": np.asarray(thresholds, dtype=float),
            "alarms": alarms,
            "init_threshold": float(self.init_threshold),
        }
