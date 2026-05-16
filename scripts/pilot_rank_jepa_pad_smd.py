import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.make_uad_smd import compute_domain_shift_metrics
from src.data.omni_smd import RawSMDMachine
from src.shift import TSJepaConfig, TSJepaModel, compute_pad_from_latents, extract_jepa_features, train_ts_jepa
from src.utils.data_paths import resolve_raw_smd_root


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)


def machine_family(name: str) -> str:
    parts = name.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else name


def list_machine_names(raw_root: Path):
    return sorted(p.stem for p in (raw_root / "train").glob("machine-*.txt"))


def to_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [T, C], got {x.shape}")
    t_steps, n_feats = x.shape
    if t_steps < window:
        pad = np.zeros((window - t_steps, n_feats), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)[None, ...]
    starts = np.arange(0, t_steps - window + 1, stride, dtype=np.int64)
    return np.stack([x[s:s + window] for s in starts], axis=0).astype(np.float32)


def labels_to_windows(y: np.ndarray, window: int, stride: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if len(y) < window:
        return np.asarray([int(y.max() > 0)], dtype=np.int64)
    starts = np.arange(0, len(y) - window + 1, stride, dtype=np.int64)
    return np.asarray([int(y[s:s + window].max() > 0) for s in starts], dtype=np.int64)


def sample_cap(x: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or len(x) <= max_items:
        return np.asarray(x)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=max_items, replace=False)
    return np.asarray(x)[idx]


def build_machine_windows(
    *,
    raw_root: Path,
    machine: str,
    preprocess_mode: str,
    window: int,
    stride: int,
) -> dict:
    data = RawSMDMachine.from_root(raw_root, machine, preprocess_mode=preprocess_mode)
    x_source = to_windows(data.x_train, window=window, stride=stride)
    y_source = np.zeros((len(x_source),), dtype=np.int64)
    x_target = to_windows(data.x_test, window=window, stride=stride)
    y_target = labels_to_windows(data.y_test, window=window, stride=stride)
    x_target_normal = x_target[y_target == 0]
    return {
        "source_windows": x_source,
        "source_labels": y_source,
        "target_windows": x_target,
        "target_labels": y_target,
        "target_normal_windows": x_target_normal,
    }


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def top_pairs_by_source(pair_rows: List[Dict]) -> List[Dict]:
    best = {}
    for row in pair_rows:
        src = row["source_machine"]
        if src not in best or row["pad_latent"]["pad_value"] > best[src]["pad_latent"]["pad_value"]:
            best[src] = row
    return [best[k] for k in sorted(best.keys())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_smd_root", default="data/ServerMachineDataset")
    ap.add_argument("--out_dir", default="outputs/benchmarks/jepa_pad_pilot")
    ap.add_argument("--checkpoint_path", default="outputs/checkpoints/ts_jepa_pilot.pt")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--preprocess_mode", default="train_zscore", choices=["train_zscore", "official_minmax"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--valid_ratio", type=float, default=0.10)
    ap.add_argument("--mask_ratio", type=float, default=0.40)
    ap.add_argument("--mask_span", type=int, default=8)
    ap.add_argument("--ema_momentum", type=float, default=0.99)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--predictor_hidden", type=int, default=128)
    ap.add_argument("--max_train_windows_per_machine", type=int, default=1000)
    ap.add_argument("--max_eval_windows_per_machine", type=int, default=1000)
    ap.add_argument("--max_pad_samples", type=int, default=1000)
    ap.add_argument("--pad_logreg_c", type=float, default=0.01)
    ap.add_argument("--pair_scope", default="same_family", choices=["same_family", "all"])
    ap.add_argument("--include_summary_baseline", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    raw_root = resolve_raw_smd_root(args.raw_smd_root)
    out_dir = Path(args.out_dir)
    checkpoint_path = Path(args.checkpoint_path)

    machine_names = list_machine_names(raw_root)
    if not machine_names:
        raise FileNotFoundError(f"No machine-*.txt files under {raw_root / 'train'}")

    print(f"[INFO] Raw root      : {raw_root}")
    print(f"[INFO] Device        : {device}")
    print(f"[INFO] Machines      : {len(machine_names)}")
    print(f"[INFO] Pair scope    : {args.pair_scope}")
    print(f"[INFO] Window/stride : {args.window}/{args.stride}")

    machine_cache = {}
    pretrain_blocks = []
    for idx, machine in enumerate(machine_names):
        pack = build_machine_windows(
            raw_root=raw_root,
            machine=machine,
            preprocess_mode=args.preprocess_mode,
            window=args.window,
            stride=args.stride,
        )
        pack["source_windows_eval"] = sample_cap(
            pack["source_windows"],
            args.max_eval_windows_per_machine,
            seed=args.seed + idx,
        )
        pack["target_normal_windows_eval"] = sample_cap(
            pack["target_normal_windows"],
            args.max_eval_windows_per_machine,
            seed=args.seed + 1000 + idx,
        )
        pretrain_blocks.append(
            sample_cap(
                pack["source_windows"],
                args.max_train_windows_per_machine,
                seed=args.seed + 2000 + idx,
            )
        )
        machine_cache[machine] = pack
        print(
            f"[CACHE] {machine}: source={len(pack['source_windows'])} "
            f"target_normal={len(pack['target_normal_windows'])}"
        )

    x_pretrain = np.concatenate(pretrain_blocks, axis=0).astype(np.float32)
    print(f"[INFO] Global JEPA pretrain windows: {x_pretrain.shape}")

    cfg = TSJepaConfig(
        in_channels=x_pretrain.shape[-1],
        d_model=args.d_model,
        predictor_hidden=args.predictor_hidden,
        mask_ratio=args.mask_ratio,
        mask_span=args.mask_span,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        valid_ratio=args.valid_ratio,
        ema_momentum=args.ema_momentum,
        seed=args.seed,
    )
    model = TSJepaModel(cfg)
    train_history = train_ts_jepa(model, x_pretrain, device=device)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "state_dict": model.state_dict(),
            "train_history": train_history,
        },
        checkpoint_path,
    )
    print(f"[OK] Saved checkpoint: {checkpoint_path}")

    latent_cache = {}
    for idx, machine in enumerate(machine_names):
        pack = machine_cache[machine]
        latent_cache[machine] = {
            "source_latent": extract_jepa_features(
                model,
                pack["source_windows_eval"],
                device=device,
                batch_size=args.batch_size,
            ),
            "target_normal_latent": extract_jepa_features(
                model,
                pack["target_normal_windows_eval"],
                device=device,
                batch_size=args.batch_size,
            ),
        }
        print(
            f"[LATENT] {machine}: source={latent_cache[machine]['source_latent'].shape} "
            f"target={latent_cache[machine]['target_normal_latent'].shape}"
        )

    pair_rows = []
    pair_index = 0
    for src in machine_names:
        src_family = machine_family(src)
        for tgt in machine_names:
            if src == tgt:
                continue
            if args.pair_scope == "same_family" and machine_family(tgt) != src_family:
                continue

            pad_metrics = compute_pad_from_latents(
                latent_cache[src]["source_latent"],
                latent_cache[tgt]["target_normal_latent"],
                seed=args.seed + pair_index,
                max_samples=args.max_pad_samples,
                logreg_c=args.pad_logreg_c,
            )
            row = {
                "source_machine": src,
                "target_machine": tgt,
                "family": src_family,
                "n_source_windows": int(len(machine_cache[src]["source_windows_eval"])),
                "n_target_normal_windows": int(len(machine_cache[tgt]["target_normal_windows_eval"])),
                "pad_latent": pad_metrics,
            }
            if args.include_summary_baseline:
                row["summary_baseline"] = compute_domain_shift_metrics(
                    machine_cache[src]["source_windows_eval"],
                    machine_cache[tgt]["target_normal_windows_eval"],
                    seed=args.seed + pair_index,
                    max_samples=args.max_pad_samples,
                )
            pair_rows.append(row)
            pair_index += 1

    pair_rows.sort(key=lambda x: x["pad_latent"]["pad_value"], reverse=True)
    top_by_source = top_pairs_by_source(pair_rows)
    payload = {
        "method": "ben_david_plus_ts_jepa_pad_pilot",
        "notes": {
            "primary_metric": "PAD_latent",
            "target_protocol": "source normal vs target normal",
            "pair_scope": args.pair_scope,
            "pretrain_encoder": "global TS-JEPA-style encoder",
        },
        "raw_smd_root": str(raw_root),
        "device": device,
        "config": cfg.to_dict(),
        "window": args.window,
        "stride": args.stride,
        "preprocess_mode": args.preprocess_mode,
        "max_train_windows_per_machine": args.max_train_windows_per_machine,
        "max_eval_windows_per_machine": args.max_eval_windows_per_machine,
        "max_pad_samples": args.max_pad_samples,
        "pad_logreg_c": args.pad_logreg_c,
        "machines": machine_names,
        "train_history": train_history,
        "machine_stats": {
            machine: {
                "source_windows": int(len(pack["source_windows"])),
                "target_windows": int(len(pack["target_windows"])),
                "target_normal_windows": int(len(pack["target_normal_windows"])),
            }
            for machine, pack in machine_cache.items()
        },
        "pair_rankings": pair_rows,
        "top_pair_overall": pair_rows[0] if pair_rows else None,
        "top_pairs_by_source": top_by_source,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "pilot_rankings.json"
    save_json(result_path, payload)
    print(f"[OK] Saved rankings: {result_path}")

    print("\n[TOP 10 PAD_latent pairs]")
    for row in pair_rows[:10]:
        print(
            f"  {row['source_machine']} -> {row['target_machine']} | "
            f"PAD={row['pad_latent']['pad_value']:.4f} | "
            f"ACC={row['pad_latent']['domain_acc']:.4f} | "
            f"AUC={row['pad_latent']['domain_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
