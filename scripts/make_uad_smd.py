import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SHIFT_RANGES = {
    "mild": (0.00, 0.20),
    "medium": (0.15, 0.45),
    "hard": (0.35, 0.75),
    "auto": (0.00, 0.75),
}


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


def window_summary_features(X: np.ndarray) -> np.ndarray:
    feat_mean = X.mean(axis=1)
    feat_std = X.std(axis=1)
    feat_min = X.min(axis=1)
    feat_max = X.max(axis=1)
    return np.concatenate([feat_mean, feat_std, feat_min, feat_max], axis=1).astype(np.float32)


def compute_domain_shift_metrics(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    seed: int = 42,
    max_samples: int = 2000,
):
    if len(X_source) == 0 or len(X_target) == 0:
        return {
            "domain_auc": float("nan"),
            "domain_acc": float("nan"),
            "feature_mean_l2": float("nan"),
            "feature_std_l2": float("nan"),
            "n_source_used": 0,
            "n_target_used": 0,
        }

    rng = np.random.RandomState(seed)
    n_src = min(len(X_source), max_samples)
    n_tgt = min(len(X_target), max_samples)
    idx_src = rng.choice(len(X_source), size=n_src, replace=False) if len(X_source) > n_src else np.arange(len(X_source))
    idx_tgt = rng.choice(len(X_target), size=n_tgt, replace=False) if len(X_target) > n_tgt else np.arange(len(X_target))

    Fs = window_summary_features(X_source[idx_src])
    Ft = window_summary_features(X_target[idx_tgt])

    mean_l2 = float(np.linalg.norm(Fs.mean(axis=0) - Ft.mean(axis=0)))
    std_l2 = float(np.linalg.norm(Fs.std(axis=0) - Ft.std(axis=0)))

    X_dom = np.concatenate([Fs, Ft], axis=0)
    y_dom = np.concatenate([np.zeros(len(Fs), dtype=int), np.ones(len(Ft), dtype=int)], axis=0)

    if len(np.unique(y_dom)) < 2 or min((y_dom == 0).sum(), (y_dom == 1).sum()) < 2:
        return {
            "domain_auc": float("nan"),
            "domain_acc": float("nan"),
            "feature_mean_l2": mean_l2,
            "feature_std_l2": std_l2,
            "n_source_used": int(len(Fs)),
            "n_target_used": int(len(Ft)),
        }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_dom,
        y_dom,
        test_size=0.30,
        random_state=seed,
        stratify=y_dom,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)

    return {
        "domain_auc": float(roc_auc_score(y_te, prob)),
        "domain_acc": float(accuracy_score(y_te, pred)),
        "feature_mean_l2": mean_l2,
        "feature_std_l2": std_l2,
        "n_source_used": int(len(Fs)),
        "n_target_used": int(len(Ft)),
    }


def infer_default_out_dir(source_dir: Path, target_dir: Path, shift_level: str) -> Path:
    same_machine = source_dir.resolve() == target_dir.resolve()
    if same_machine:
        root = source_dir.parent.parent / "smd_experiments" / f"temporal_{shift_level}"
        return root / source_dir.name

    root = source_dir.parent.parent / "smd_experiments" / f"cross_machine_{shift_level}"
    return root / f"{source_dir.name}__to__{target_dir.name}"


def target_split_from_start(
    Xt: np.ndarray,
    yt: np.ndarray,
    *,
    pool_start: int,
    n_pool: int,
    val_frac: float,
    guard: int,
    min_val: int,
    min_test: int,
):
    n_t = len(Xt)
    pool_end = min(n_t, pool_start + n_pool)
    val_start = min(n_t, pool_end + guard)
    remaining_after_pool = max(0, n_t - val_start)

    if remaining_after_pool <= 0:
        return None

    min_val_eff = max(1, min_val)
    desired_val = int(np.floor(remaining_after_pool * val_frac))
    desired_val = max(min_val_eff, desired_val)
    max_val = remaining_after_pool - guard - min_test
    if max_val < min_val_eff:
        return None

    n_val = min(desired_val, max_val)
    val_end = min(n_t, val_start + n_val)
    test_start = min(n_t, val_end + guard)
    test_end = n_t

    X_pool = Xt[pool_start:pool_end]
    y_pool = yt[pool_start:pool_end]
    X_val = Xt[val_start:val_end]
    y_val = yt[val_start:val_end]
    X_test = Xt[test_start:test_end]
    y_test = yt[test_start:test_end]

    return {
        "pool_start": int(pool_start),
        "pool_end": int(pool_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
        "test_start": int(test_start),
        "test_end": int(test_end),
        "X_pool": X_pool,
        "y_pool": y_pool,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def split_is_valid(
    split: dict,
    *,
    min_target_pool: int,
    min_val: int,
    min_test: int,
    min_anom_val: int,
    min_anom_test: int,
    max_pool_anom_ratio: float,
    require_both_classes: bool,
):
    X_pool = split["X_pool"]
    y_pool = split["y_pool"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    if len(X_pool) < min_target_pool or len(X_val) < min_val or len(X_test) < min_test:
        return False

    pool_anom_ratio = float(y_pool.mean()) if len(y_pool) > 0 else 1.0
    if pool_anom_ratio > max_pool_anom_ratio:
        return False

    if int((y_val == 1).sum()) < min_anom_val:
        return False
    if int((y_test == 1).sum()) < min_anom_test:
        return False

    if require_both_classes:
        if np.unique(y_val).size < 2:
            return False
        if np.unique(y_test).size < 2:
            return False

    return True


def candidate_pool_starts(
    n_t: int,
    *,
    n_pool: int,
    guard: int,
    min_val: int,
    min_test: int,
    shift_level: str,
    search_step: int,
):
    low_frac, high_frac = SHIFT_RANGES[shift_level]
    max_start = max(0, n_t - n_pool - guard - max(1, min_val) - guard - max(1, min_test))
    low = min(max_start, int(np.floor(low_frac * n_t)))
    high = min(max_start, int(np.floor(high_frac * n_t)))

    starts = list(range(low, high + 1, max(1, search_step)))
    if low not in starts:
        starts.append(low)
    if high not in starts:
        starts.append(high)
    if 0 <= max_start and max_start not in starts and shift_level == "auto":
        starts.append(max_start)
    starts = sorted({int(s) for s in starts if 0 <= s <= max_start})
    return starts


def search_best_split(
    Xs_norm: np.ndarray,
    Xt: np.ndarray,
    yt: np.ndarray,
    *,
    target_pool_frac: float,
    val_frac: float,
    guard: int,
    min_target_pool: int,
    min_val: int,
    min_test: int,
    min_anom_val: int,
    min_anom_test: int,
    max_pool_anom_ratio: float,
    require_both_classes: bool,
    shift_level: str,
    search_step: int,
    seed: int,
):
    n_t = len(Xt)
    n_pool = max(max(1, min_target_pool), int(np.floor(n_t * target_pool_frac)))
    starts = candidate_pool_starts(
        n_t,
        n_pool=n_pool,
        guard=guard,
        min_val=min_val,
        min_test=min_test,
        shift_level=shift_level,
        search_step=search_step,
    )

    best = None
    best_key = None

    for pool_start in starts:
        split = target_split_from_start(
            Xt,
            yt,
            pool_start=pool_start,
            n_pool=n_pool,
            val_frac=val_frac,
            guard=guard,
            min_val=min_val,
            min_test=min_test,
        )
        if split is None:
            continue

        if not split_is_valid(
            split,
            min_target_pool=min_target_pool,
            min_val=min_val,
            min_test=min_test,
            min_anom_val=min_anom_val,
            min_anom_test=min_anom_test,
            max_pool_anom_ratio=max_pool_anom_ratio,
            require_both_classes=require_both_classes,
        ):
            continue

        shift_pool = compute_domain_shift_metrics(Xs_norm, split["X_pool"], seed=seed)
        shift_val = compute_domain_shift_metrics(Xs_norm, split["X_val"], seed=seed)
        shift_test = compute_domain_shift_metrics(Xs_norm, split["X_test"], seed=seed)

        pool_anom_ratio = float(split["y_pool"].mean()) if len(split["y_pool"]) else 0.0
        val_anom = int((split["y_val"] == 1).sum())
        test_anom = int((split["y_test"] == 1).sum())
        rank_auc = shift_pool["domain_auc"]
        if not np.isfinite(rank_auc):
            rank_auc = 0.5

        rank_key = (
            float(rank_auc),
            -float(pool_anom_ratio),
            int(min(val_anom, test_anom)),
            int(len(split["X_test"])),
        )

        split["shift_pool"] = shift_pool
        split["shift_val"] = shift_val
        split["shift_test"] = shift_test
        split["pool_anom_ratio"] = pool_anom_ratio
        split["val_anom_count"] = val_anom
        split["test_anom_count"] = test_anom

        if best is None or rank_key > best_key:
            best = split
            best_key = rank_key

    return best


def fixed_split(
    Xt: np.ndarray,
    yt: np.ndarray,
    *,
    target_pool_frac: float,
    val_frac: float,
    guard: int,
):
    n_pool = max(1, int(np.floor(len(Xt) * target_pool_frac)))
    return target_split_from_start(
        Xt,
        yt,
        pool_start=0,
        n_pool=n_pool,
        val_frac=val_frac,
        guard=guard,
        min_val=1,
        min_test=1,
    )


def build_metadata(
    *,
    source_dir: Path,
    target_dir: Path,
    out_dir: Path,
    split: dict,
    shift_level: str,
    split_mode: str,
    train_count: int,
    train_frac: float,
):
    y_val = split["y_val"]
    y_test = split["y_test"]
    y_pool = split["y_pool"]

    return {
        "protocol_version": 2,
        "split_mode": split_mode,
        "shift_level": shift_level,
        "source_machine": source_dir.name,
        "target_machine": target_dir.name,
        "same_machine": bool(source_dir.resolve() == target_dir.resolve()),
        "out_dir": str(out_dir),
        "train_normal_count": int(train_count),
        "train_normal_frac": float(train_frac),
        "target_pool_count": int(len(split["X_pool"])),
        "target_pool_hidden_anomaly_count": int((y_pool == 1).sum()),
        "target_pool_hidden_anomaly_ratio": float(split["pool_anom_ratio"]),
        "val_count": int(len(split["X_val"])),
        "val_normal_count": int((y_val == 0).sum()),
        "val_anomaly_count": int((y_val == 1).sum()),
        "test_count": int(len(split["X_test"])),
        "test_normal_count": int((y_test == 0).sum()),
        "test_anomaly_count": int((y_test == 1).sum()),
        "indices": {
            "pool_start": int(split["pool_start"]),
            "pool_end": int(split["pool_end"]),
            "val_start": int(split["val_start"]),
            "val_end": int(split["val_end"]),
            "test_start": int(split["test_start"]),
            "test_end": int(split["test_end"]),
        },
        "domain_shift": {
            "source_vs_target_pool": split["shift_pool"],
            "source_vs_val": split["shift_val"],
            "source_vs_test": split["shift_test"],
        },
        "notes": (
            "No synthetic data was created. train_normal comes from source.npz of the source machine, "
            "while target_pool/val/test are contiguous windows from target.npz of the target machine."
        ),
    }


def create_dataset(args):
    source_dir = Path(args.machine_dir)
    target_dir = Path(args.target_machine_dir) if args.target_machine_dir else source_dir

    src_path = source_dir / args.source_name
    tgt_path = target_dir / args.target_name
    out_dir = Path(args.out_dir) if args.out_dir else infer_default_out_dir(source_dir, target_dir, args.shift_level)

    Xs, ys = load_npz(str(src_path))
    Xt, yt = load_npz(str(tgt_path))

    if ys is None:
        raise ValueError(f"{src_path} must contain y to filter normal for train_normal.")
    if yt is None:
        raise ValueError(f"{tgt_path} must contain y to create target_pool/val/test.")

    ys = binarize_y(ys)
    yt = binarize_y(yt)

    idx_norm = np.where(ys == 0)[0]
    if len(idx_norm) == 0:
        raise ValueError("No normal samples (y==0) found in source.")

    Xs_norm = Xs[idx_norm]
    train_frac = max(0.0, min(1.0, float(args.train_normal_frac)))
    n_train = max(1, int(np.floor(len(Xs_norm) * train_frac)))
    X_train = Xs_norm[:n_train]
    y_train = np.zeros(len(X_train), dtype=int)

    if args.split_mode == "fixed":
        split = fixed_split(
            Xt,
            yt,
            target_pool_frac=args.target_pool_frac,
            val_frac=args.val_frac,
            guard=args.guard,
        )
        if split is None:
            raise ValueError("Could not create split with split_mode=fixed.")
        shift_pool = compute_domain_shift_metrics(X_train, split["X_pool"], seed=args.seed)
        shift_val = compute_domain_shift_metrics(X_train, split["X_val"], seed=args.seed)
        shift_test = compute_domain_shift_metrics(X_train, split["X_test"], seed=args.seed)
        split["shift_pool"] = shift_pool
        split["shift_val"] = shift_val
        split["shift_test"] = shift_test
        split["pool_anom_ratio"] = float(split["y_pool"].mean()) if len(split["y_pool"]) else 0.0
    else:
        split = search_best_split(
            X_train,
            Xt,
            yt,
            target_pool_frac=args.target_pool_frac,
            val_frac=args.val_frac,
            guard=args.guard,
            min_target_pool=args.min_target_pool,
            min_val=args.min_val,
            min_test=args.min_test,
            min_anom_val=args.min_anom_val,
            min_anom_test=args.min_anom_test,
            max_pool_anom_ratio=args.max_pool_anom_ratio,
            require_both_classes=not args.allow_single_class_eval,
            shift_level=args.shift_level,
            search_step=args.search_step,
            seed=args.seed,
        )
        if split is None:
            raise ValueError(
                "No valid split found. Try relaxing constraints, reducing guard, lowering min_anom_val/min_anom_test, "
                "or switching shift_level."
            )

    if args.min_train and len(X_train) < args.min_train:
        warn_or_raise(f"train_normal small: {len(X_train)} < min_train={args.min_train}", args.strict)

    metadata = build_metadata(
        source_dir=source_dir,
        target_dir=target_dir,
        out_dir=out_dir,
        split=split,
        shift_level=args.shift_level,
        split_mode=args.split_mode,
        train_count=len(X_train),
        train_frac=train_frac,
    )

    out_train = out_dir / args.out_train
    out_target_pool = out_dir / args.out_target_pool
    out_val = out_dir / args.out_val
    out_test = out_dir / args.out_test
    out_meta = out_dir / args.out_meta

    save_npz(str(out_train), X_train, y_train)
    save_npz(str(out_target_pool), split["X_pool"], None)
    save_npz(str(out_val), split["X_val"], split["y_val"])
    save_npz(str(out_test), split["X_test"], split["y_test"])
    os.makedirs(out_dir, exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[DONE] Created experiment dataset:")
    print(f"  source machine       : {source_dir.name}")
    print(f"  target machine       : {target_dir.name}")
    print(f"  out_dir              : {out_dir}")
    print(f"  train_normal         : X={X_train.shape} y=(all 0)")
    print(
        f"  target_pool_unlabeled: X={split['X_pool'].shape} "
        f"hidden_anom_ratio={metadata['target_pool_hidden_anomaly_ratio']:.4f}"
    )
    print(
        f"  val_mixed            : X={split['X_val'].shape} "
        f"normal={metadata['val_normal_count']} anom={metadata['val_anomaly_count']}"
    )
    print(
        f"  test_mixed           : X={split['X_test'].shape} "
        f"normal={metadata['test_normal_count']} anom={metadata['test_anomaly_count']}"
    )
    print(
        "  shift(source->pool)  : "
        f"domain_auc={metadata['domain_shift']['source_vs_target_pool']['domain_auc']:.4f} "
        f"domain_acc={metadata['domain_shift']['source_vs_target_pool']['domain_acc']:.4f} "
        f"mean_l2={metadata['domain_shift']['source_vs_target_pool']['feature_mean_l2']:.4f}"
    )
    if args.guard > 0:
        print(f"  guard                : {args.guard} windows")

    return metadata


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--machine_dir",
        required=True,
        help="Source machine directory, e.g. data/smd/machine-1-1",
    )
    ap.add_argument(
        "--target_machine_dir",
        default=None,
        help="Optional target machine directory for cross-machine domain shift. If omitted, use the same machine.",
    )
    ap.add_argument("--source_name", default="source.npz")
    ap.add_argument("--target_name", default="target.npz")

    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--out_train", default="train_normal.npz")
    ap.add_argument("--out_target_pool", default="target_pool_unlabeled.npz")
    ap.add_argument("--out_val", default="val_mixed.npz")
    ap.add_argument("--out_test", default="test_mixed.npz")
    ap.add_argument("--out_meta", default="split_metadata.json")

    ap.add_argument("--split_mode", choices=["search", "fixed"], default="search")
    ap.add_argument("--shift_level", choices=sorted(SHIFT_RANGES.keys()), default="medium")
    ap.add_argument("--train_normal_frac", type=float, default=1.0)
    ap.add_argument("--target_pool_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.30)
    ap.add_argument("--guard", type=int, default=0)
    ap.add_argument("--search_step", type=int, default=4)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=0.10)

    ap.add_argument("--min_train", type=int, default=0)
    ap.add_argument("--min_target_pool", type=int, default=32)
    ap.add_argument("--min_val", type=int, default=32)
    ap.add_argument("--min_test", type=int, default=64)
    ap.add_argument("--min_anom_val", type=int, default=3)
    ap.add_argument("--min_anom_test", type=int, default=5)
    ap.add_argument("--allow_single_class_eval", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true")
    return ap


def main():
    args = build_arg_parser().parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
