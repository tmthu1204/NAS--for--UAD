import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from src.data.omni_smd import contiguous_train_valid_split
from src.data.tranad_smd import build_tranad_windows, load_raw_tranad_smd_machine
from src.families.tranad import (
    TranADModel,
    get_fixed_paper_tranad_arch,
    score_tranad_windows,
    train_tranad_source,
    validate_tranad_on_windows,
)
from src.utils.metrics import (
    best_f1,
    compute_ap_auroc,
    event_f1_and_delay,
    f1_at_threshold,
    pot_threshold,
)


LOGS = os.path.join(PROJ, "outputs", "logs")
BENCHMARKS = os.path.join(PROJ, "outputs", "benchmarks")
os.makedirs(LOGS, exist_ok=True)
os.makedirs(BENCHMARKS, exist_ok=True)


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


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)


def compute_metrics(y_eval, scores_eval, scores_train, *, pot_q, pot_level):
    ap, auroc = compute_ap_auroc(y_eval, scores_eval)
    thr_pot = pot_threshold(scores_train, q=pot_q, level=pot_level)
    p_pot, r_pot, f1_pot = f1_at_threshold(y_eval, scores_eval, thr_pot)
    f1_b, p_b, r_b, thr_b = best_f1(y_eval, scores_eval)
    y_pred_bin = (scores_eval >= float(thr_pot)).astype(int)
    ev = event_f1_and_delay(y_eval, y_pred_bin)
    return {
        "ap": float(ap),
        "auroc": float(auroc),
        "f1_pot": float(f1_pot),
        "precision_pot": float(p_pot),
        "recall_pot": float(r_pot),
        "thr_pot": float(thr_pot),
        "f1_best": float(f1_b),
        "precision_best": float(p_b),
        "recall_best": float(r_b),
        "thr_best": float(thr_b),
        "event_f1": float(ev["event_f1"]),
        "event_precision": float(ev["event_precision"]),
        "event_recall": float(ev["event_recall"]),
        "delay_mean": float(ev["delay_mean"]),
        "delay_median": float(ev["delay_median"]),
    }


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_smd_root", default=os.path.join(PROJ, "data", "ServerMachineDataset"))
    ap.add_argument("--machine", default="machine-1-1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--final_epochs", type=int, default=5)
    ap.add_argument("--window_length", type=int, default=10)
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_start", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--test_start", type=int, default=0)
    ap.add_argument("--test_limit", type=int, default=0)
    ap.add_argument("--pot_q", type=float, default=1e-3)
    ap.add_argument("--pot_level", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_global_seed(args.seed)

    x_train, x_test, y_test = load_raw_tranad_smd_machine(args.raw_smd_root, args.machine)
    if args.train_limit > 0:
        x_train = x_train[args.train_start:args.train_start + args.train_limit]
    elif args.train_start > 0:
        x_train = x_train[args.train_start:]
    if args.test_limit > 0:
        x_test = x_test[args.test_start:args.test_start + args.test_limit]
        y_test = y_test[args.test_start:args.test_start + args.test_limit]
    elif args.test_start > 0:
        x_test = x_test[args.test_start:]
        y_test = y_test[args.test_start:]

    arch = get_fixed_paper_tranad_arch(window_length=args.window_length)
    arch.batch_size = args.batch_size
    arch.max_epoch = args.epochs
    arch.valid_ratio = args.valid_ratio

    train_windows_full = build_tranad_windows(x_train, arch.window_length)
    train_windows_inner, val_windows_inner = contiguous_train_valid_split(
        train_windows_full,
        valid_ratio=arch.valid_ratio,
    )
    test_windows = build_tranad_windows(x_test, arch.window_length)

    model = TranADModel(x_train.shape[1], arch).to(args.device).double()
    search_log = train_tranad_source(
        model,
        train_windows_inner,
        val_windows_inner,
        device=args.device,
        arch=arch,
        epochs=args.epochs,
        shuffle=False,
        use_early_stopping=False,
        restore_best_state=False,
    )
    val_stats = validate_tranad_on_windows(
        model,
        val_windows_inner,
        device=args.device,
        batch_size=arch.batch_size,
        epoch_idx=args.epochs,
    )

    final_model = TranADModel(x_train.shape[1], arch).to(args.device).double()
    final_model.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    final_log = train_tranad_source(
        final_model,
        train_windows_full,
        None,
        device=args.device,
        arch=arch,
        epochs=args.final_epochs,
        shuffle=False,
        use_early_stopping=False,
        restore_best_state=False,
    )

    scores_train = score_tranad_windows(final_model, train_windows_full, device=args.device, batch_size=arch.batch_size)
    scores_test = score_tranad_windows(final_model, test_windows, device=args.device, batch_size=arch.batch_size)
    metrics_uad = compute_metrics(y_test, scores_test, scores_train, pot_q=args.pot_q, pot_level=args.pot_level)

    result = {
        "mode": "uad_source",
        "family": "tranad_upstream",
        "protocol": "raw_smd_machine_by_machine",
        "machine": args.machine,
        "raw_smd_root": str(args.raw_smd_root),
        "arch": str(arch),
        "val_stats": val_stats,
        "search_train_curve": search_log,
        "final_train_curve": final_log,
        "metrics_uad": metrics_uad,
        "tranad_upstream_notes": {
            "reference_repo": "imperial-qore/TranAD",
            "preprocess_mode": "none",
            "window_protocol": "front_padded_history_windows",
            "core": "fixed_upstream_architecture",
        },
    }

    bench_name = "tranad-upstream"
    if args.tag.strip():
        bench_name = f"{bench_name}-{args.tag.strip()}"
    bench_dir = os.path.join(BENCHMARKS, bench_name)
    save_json(os.path.join(bench_dir, f"{args.machine}.json"), result)
    print(f"[OK] Saved benchmark result to {os.path.relpath(os.path.join(bench_dir, f'{args.machine}.json'), PROJ)}")


if __name__ == "__main__":
    main()
