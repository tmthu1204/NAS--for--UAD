import argparse
import importlib
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.utils.data as data_utils


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
LOGS = os.path.join(PROJ, "outputs", "logs")
BENCHMARKS = os.path.join(PROJ, "outputs", "benchmarks")
UPSTREAM = os.path.join(PROJ, "external", "usad_upstream")
os.makedirs(LOGS, exist_ok=True)
os.makedirs(BENCHMARKS, exist_ok=True)

sys.path.insert(0, PROJ)
sys.path.insert(0, UPSTREAM)

try:
    import seaborn  # type: ignore  # noqa: F401
except Exception:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

from src.data.omni_smd import aligned_last_point_labels, contiguous_train_valid_split
from src.data.swat import (
    RawSWaTDataset,
    build_upstream_usad_flat_windows,
    build_upstream_usad_window_labels,
)
from src.utils.metrics import (
    best_f1,
    compute_ap_auroc,
    event_f1_and_delay,
    f1_at_threshold,
    pot_threshold,
)


def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)


def _concat_upstream_scores(results) -> np.ndarray:
    parts = []
    for tensor in results:
        arr = tensor.detach().cpu().numpy().reshape(-1)
        parts.append(arr)
    if not parts:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _metrics_from_scores(y_eval, scores_eval, scores_train, *, pot_q=1e-3, pot_level=0.99):
    y_eval = np.asarray(y_eval).astype(int)
    scores_eval = np.asarray(scores_eval).astype(float)
    scores_train = np.asarray(scores_train).astype(float)

    ap, auroc = compute_ap_auroc(y_eval, scores_eval)
    f1_best, p_best, r_best, thr_best = best_f1(y_eval, scores_eval)

    thr_pot = pot_threshold(scores_train, q=pot_q, level=pot_level)
    p_pot, r_pot, f1_pot = f1_at_threshold(y_eval, scores_eval, thr_pot)
    y_pred_bin = (scores_eval >= thr_pot).astype(int)
    ev = event_f1_and_delay(y_eval, y_pred_bin)

    return {
        "ap": float(ap),
        "auroc": float(auroc),
        "f1_pot": float(f1_pot),
        "precision_pot": float(p_pot),
        "recall_pot": float(r_pot),
        "thr_pot": float(thr_pot),
        "f1_best": float(f1_best),
        "precision_best": float(p_best),
        "recall_best": float(r_best),
        "thr_best": float(thr_best),
        "event_f1": float(ev["event_f1"]),
        "event_precision": float(ev["event_precision"]),
        "event_recall": float(ev["event_recall"]),
        "delay_mean": float(ev["delay_mean"]),
        "delay_median": float(ev["delay_median"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=os.path.join(PROJ, "data", "SWaT", "normal.csv"))
    ap.add_argument("--test_csv", default=os.path.join(PROJ, "data", "SWaT", "attack.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--window_length", type=int, default=12)
    ap.add_argument("--hidden_size", type=int, default=100)
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--train_limit", type=int, default=12000)
    ap.add_argument("--test_limit", type=int, default=12000)
    ap.add_argument("--preprocess", default="train_minmax", choices=["train_minmax", "train_zscore"])
    ap.add_argument("--pot_q", type=float, default=1e-3)
    ap.add_argument("--pot_level", type=float, default=0.99)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not os.path.isdir(UPSTREAM):
        raise FileNotFoundError(f"Upstream USAD repo not found: {UPSTREAM}")

    usad_mod = importlib.import_module("usad")
    usad_mod.device = torch.device(args.device)

    swat_data = RawSWaTDataset.from_csvs(
        args.train_csv,
        args.test_csv,
        preprocess_mode=args.preprocess,
        downsample=args.downsample,
    )
    if args.train_limit > 0:
        swat_data.x_train = swat_data.x_train[:args.train_limit]
    if args.test_limit > 0:
        swat_data.x_test = swat_data.x_test[:args.test_limit]
        swat_data.y_test = swat_data.y_test[:args.test_limit]

    train_windows_full = build_upstream_usad_flat_windows(
        swat_data.x_train,
        args.window_length,
        args.stride,
    )
    x_train_inner, x_val_inner = contiguous_train_valid_split(
        train_windows_full,
        valid_ratio=args.valid_ratio,
    )
    y_test_aligned = build_upstream_usad_window_labels(
        swat_data.y_test,
        window_length=args.window_length,
        stride=args.stride,
    )
    test_windows = build_upstream_usad_flat_windows(
        swat_data.x_test,
        args.window_length,
        args.stride,
    )

    train_tensor = torch.from_numpy(np.ascontiguousarray(x_train_inner))
    val_tensor = torch.from_numpy(np.ascontiguousarray(x_val_inner))
    full_train_tensor = torch.from_numpy(np.ascontiguousarray(train_windows_full))
    test_tensor = torch.from_numpy(np.ascontiguousarray(test_windows))

    w_size = int(train_tensor.shape[1])
    z_size = int(args.window_length * args.hidden_size)

    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(train_tensor),
        batch_size=min(args.batch_size, max(1, len(train_tensor))),
        shuffle=False,
        num_workers=0,
    )
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(val_tensor),
        batch_size=min(args.batch_size, max(1, len(val_tensor))),
        shuffle=False,
        num_workers=0,
    )
    full_train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(full_train_tensor),
        batch_size=min(args.batch_size, max(1, len(full_train_tensor))),
        shuffle=False,
        num_workers=0,
    )
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(test_tensor),
        batch_size=min(args.batch_size, max(1, len(test_tensor))),
        shuffle=False,
        num_workers=0,
    )

    model = usad_mod.UsadModel(w_size, z_size)
    model = usad_mod.to_device(model, usad_mod.device)
    search_history = usad_mod.training(args.epochs, model, train_loader, val_loader)

    final_model = usad_mod.UsadModel(w_size, z_size)
    final_model.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    final_model = usad_mod.to_device(final_model, usad_mod.device)
    final_history = usad_mod.training(args.epochs, final_model, full_train_loader, val_loader)

    scores_train = _concat_upstream_scores(usad_mod.testing(final_model, full_train_loader))
    scores_test = _concat_upstream_scores(usad_mod.testing(final_model, test_loader))
    metrics_uad = _metrics_from_scores(
        y_test_aligned,
        scores_test,
        scores_train,
        pot_q=args.pot_q,
        pot_level=args.pot_level,
    )

    bench_name = "usad-upstream"
    if args.tag.strip():
        bench_name = f"{bench_name}-{args.tag.strip()}"
    bench_dir = os.path.join(BENCHMARKS, bench_name)

    payload = {
        "family": "usad_upstream",
        "protocol": "raw_swat_normal_attack_subset",
        "repo_dir": UPSTREAM,
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "subset": {
            "downsample": args.downsample,
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "preprocess": args.preprocess,
            "window_length": args.window_length,
            "stride": args.stride,
            "valid_ratio": args.valid_ratio,
        },
        "upstream_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "w_size": w_size,
            "z_size": z_size,
        },
        "search_train_curve": search_history,
        "final_train_curve": final_history,
        "metrics_uad": metrics_uad,
    }

    _save_json(os.path.join(bench_dir, "swat.json"), payload)
    print(json.dumps(_sanitize_json(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
