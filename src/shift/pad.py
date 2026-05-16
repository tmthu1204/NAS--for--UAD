from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PadMetrics:
    pad_value: float
    domain_acc: float
    domain_auc: float
    feature_mean_l2: float
    n_source_used: int
    n_target_used: int

    def to_dict(self) -> dict:
        return {
            "pad_value": float(self.pad_value),
            "domain_acc": float(self.domain_acc),
            "domain_auc": float(self.domain_auc),
            "feature_mean_l2": float(self.feature_mean_l2),
            "n_source_used": int(self.n_source_used),
            "n_target_used": int(self.n_target_used),
        }


def proxy_a_distance_from_accuracy(domain_acc: float) -> float:
    if not np.isfinite(domain_acc):
        return float("nan")
    return float(np.clip(4.0 * float(domain_acc) - 2.0, 0.0, 2.0))


def compute_pad_from_latents(
    z_source: np.ndarray,
    z_target: np.ndarray,
    *,
    seed: int = 42,
    max_samples: int = 2000,
    test_size: float = 0.30,
    logreg_c: float = 0.01,
) -> dict:
    z_source = np.asarray(z_source, dtype=np.float32)
    z_target = np.asarray(z_target, dtype=np.float32)

    if z_source.ndim != 2 or z_target.ndim != 2:
        raise ValueError(
            f"Expected 2-D latent arrays, got source={z_source.shape}, target={z_target.shape}"
        )

    if len(z_source) == 0 or len(z_target) == 0:
        return PadMetrics(
            pad_value=float("nan"),
            domain_acc=float("nan"),
            domain_auc=float("nan"),
            feature_mean_l2=float("nan"),
            n_source_used=0,
            n_target_used=0,
        ).to_dict()

    rng = np.random.RandomState(seed)
    n_balanced = min(len(z_source), len(z_target), max_samples)
    if n_balanced < 4:
        mean_l2 = float(np.linalg.norm(z_source.mean(axis=0) - z_target.mean(axis=0)))
        return PadMetrics(
            pad_value=float("nan"),
            domain_acc=float("nan"),
            domain_auc=float("nan"),
            feature_mean_l2=mean_l2,
            n_source_used=int(min(len(z_source), n_balanced)),
            n_target_used=int(min(len(z_target), n_balanced)),
        ).to_dict()

    idx_source = (
        rng.choice(len(z_source), size=n_balanced, replace=False)
        if len(z_source) > n_balanced
        else np.arange(len(z_source))
    )
    idx_target = (
        rng.choice(len(z_target), size=n_balanced, replace=False)
        if len(z_target) > n_balanced
        else np.arange(len(z_target))
    )

    fs = z_source[idx_source]
    ft = z_target[idx_target]
    mean_l2 = float(np.linalg.norm(fs.mean(axis=0) - ft.mean(axis=0)))

    x_dom = np.concatenate([fs, ft], axis=0)
    y_dom = np.concatenate(
        [np.zeros(len(fs), dtype=np.int64), np.ones(len(ft), dtype=np.int64)],
        axis=0,
    )

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_dom,
        y_dom,
        test_size=test_size,
        random_state=seed,
        stratify=y_dom,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, C=logreg_c),
    )
    clf.fit(x_tr, y_tr)
    prob = clf.predict_proba(x_te)[:, 1]
    pred = clf.predict(x_te)

    domain_acc = float(accuracy_score(y_te, pred))
    domain_auc = float(roc_auc_score(y_te, prob))
    pad_value = proxy_a_distance_from_accuracy(domain_acc)

    return PadMetrics(
        pad_value=pad_value,
        domain_acc=domain_acc,
        domain_auc=domain_auc,
        feature_mean_l2=mean_l2,
        n_source_used=int(len(fs)),
        n_target_used=int(len(ft)),
    ).to_dict()
