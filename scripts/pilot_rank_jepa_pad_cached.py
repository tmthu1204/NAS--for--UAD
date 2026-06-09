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

from scripts.make_uad_smd import binarize_y, compute_domain_shift_metrics, load_npz
from src.shift import TSJepaConfig, TSJepaModel, compute_pad_from_latents, extract_jepa_features, train_ts_jepa


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


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def sample_cap(x: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or len(x) <= max_items:
        return np.asarray(x)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=max_items, replace=False)
    return np.asarray(x)[idx]


def list_entity_dirs(data_root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in data_root.iterdir()
            if p.is_dir() and (p / "source.npz").exists() and (p / "target.npz").exists()
        ]
    )


def load_cached_entity(entity_dir: Path, source_name: str, target_name: str) -> Dict[str, np.ndarray]:
    x_source, y_source = load_npz(str(entity_dir / source_name))
    y_source = binarize_y(y_source)
    if y_source is None:
        y_source = np.zeros((len(x_source),), dtype=np.int64)
    x_source_normal = x_source[y_source == 0]

    x_target, y_target = load_npz(str(entity_dir / target_name))
    y_target = binarize_y(y_target)
    if y_target is None:
        y_target = np.zeros((len(x_target),), dtype=np.int64)
    x_target_normal = x_target[y_target == 0]

    return {
        "source_windows": np.asarray(x_source, dtype=np.float32),
        "source_normal_windows": np.asarray(x_source_normal, dtype=np.float32),
        "target_windows": np.asarray(x_target, dtype=np.float32),
        "target_labels": np.asarray(y_target, dtype=np.int64),
        "target_normal_windows": np.asarray(x_target_normal, dtype=np.float32),
    }


def top_entities(rows: List[Dict], topk: int) -> List[Dict]:
    if topk <= 0:
        return rows
    return rows[:topk]


def channel_family(name: str) -> str:
    return name.split("-")[0] if "-" in name else name


def top_pairs_by_source(pair_rows: List[Dict]) -> List[Dict]:
    best = {}
    for row in pair_rows:
        src = row["source_entity"]
        if src not in best or row["pad_latent"]["pad_value"] > best[src]["pad_latent"]["pad_value"]:
            best[src] = row
    return [best[k] for k in sorted(best.keys())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Cached dataset root, e.g. data/smap or data/msl")
    ap.add_argument("--dataset_name", default="", help="Optional display name for reporting.")
    ap.add_argument("--channels", default="", help="Optional comma-separated entity ids to include.")
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--source_name", default="source.npz")
    ap.add_argument("--target_name", default="target.npz")
    ap.add_argument(
        "--pair_scope",
        default="same_entity",
        choices=["same_entity", "cross_entity_same_prefix", "cross_entity_all"],
        help="How to define source-target ranking pairs.",
    )
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--checkpoint_path", default="")
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
    ap.add_argument("--max_train_windows_per_entity", type=int, default=1000)
    ap.add_argument("--max_eval_windows_per_entity", type=int, default=1000)
    ap.add_argument("--max_pad_samples", type=int, default=1000)
    ap.add_argument("--pad_logreg_c", type=float, default=0.01)
    ap.add_argument("--include_summary_baseline", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    dataset_name = args.dataset_name.strip() or data_root.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "benchmarks" / f"jepa_pad_{dataset_name}"
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path("outputs") / "checkpoints" / f"ts_jepa_{dataset_name}.pt"
    )
    device = resolve_device(args.device)

    entity_dirs = list_entity_dirs(data_root)
    if args.channels.strip():
        keep = {c.strip() for c in args.channels.split(",") if c.strip()}
        entity_dirs = [p for p in entity_dirs if p.name in keep]
    if args.max_entities > 0:
        entity_dirs = entity_dirs[: args.max_entities]
    if not entity_dirs:
        raise ValueError(f"No cached entities found under {data_root}")

    print(f"[INFO] Cached root    : {data_root}")
    print(f"[INFO] Dataset        : {dataset_name}")
    print(f"[INFO] Device         : {device}")
    print(f"[INFO] Entities       : {len(entity_dirs)}")

    entity_cache = {}
    pretrain_blocks = []
    for idx, entity_dir in enumerate(entity_dirs):
        pack = load_cached_entity(entity_dir, args.source_name, args.target_name)
        pack["source_normal_eval"] = sample_cap(
            pack["source_normal_windows"],
            args.max_eval_windows_per_entity,
            seed=args.seed + idx,
        )
        pack["target_normal_eval"] = sample_cap(
            pack["target_normal_windows"],
            args.max_eval_windows_per_entity,
            seed=args.seed + 1000 + idx,
        )
        pretrain_blocks.append(
            sample_cap(
                pack["source_normal_windows"],
                args.max_train_windows_per_entity,
                seed=args.seed + 2000 + idx,
            )
        )
        entity_cache[entity_dir.name] = pack
        print(
            f"[CACHE] {entity_dir.name}: "
            f"source_normal={len(pack['source_normal_windows'])} "
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
    for entity_name, pack in entity_cache.items():
        latent_cache[entity_name] = {
            "source_latent": extract_jepa_features(
                model,
                pack["source_normal_eval"],
                device=device,
                batch_size=args.batch_size,
            ),
            "target_normal_latent": extract_jepa_features(
                model,
                pack["target_normal_eval"],
                device=device,
                batch_size=args.batch_size,
            ),
        }
        print(
            f"[LATENT] {entity_name}: "
            f"source={latent_cache[entity_name]['source_latent'].shape} "
            f"target={latent_cache[entity_name]['target_normal_latent'].shape}"
        )

    rows = []
    entity_names = sorted(entity_cache.keys())
    if args.pair_scope == "same_entity":
        for idx, entity_name in enumerate(entity_names):
            pack = entity_cache[entity_name]
            pad_metrics = compute_pad_from_latents(
                latent_cache[entity_name]["source_latent"],
                latent_cache[entity_name]["target_normal_latent"],
                seed=args.seed + idx,
                max_samples=args.max_pad_samples,
                logreg_c=args.pad_logreg_c,
            )
            row = {
                "entity": entity_name,
                "n_source_normal_windows": int(len(pack["source_normal_eval"])),
                "n_target_normal_windows": int(len(pack["target_normal_eval"])),
                "pad_latent": pad_metrics,
            }
            if args.include_summary_baseline:
                row["summary_baseline"] = compute_domain_shift_metrics(
                    pack["source_normal_eval"],
                    pack["target_normal_eval"],
                    seed=args.seed + idx,
                    max_samples=args.max_pad_samples,
                )
            rows.append(row)
        rows.sort(key=lambda x: x["pad_latent"]["pad_value"], reverse=True)
    else:
        same_prefix_only = args.pair_scope == "cross_entity_same_prefix"
        pair_index = 0
        for src in entity_names:
            src_family = channel_family(src)
            for tgt in entity_names:
                if src == tgt:
                    continue
                if same_prefix_only and channel_family(tgt) != src_family:
                    continue
                src_pack = entity_cache[src]
                tgt_pack = entity_cache[tgt]
                pad_metrics = compute_pad_from_latents(
                    latent_cache[src]["source_latent"],
                    latent_cache[tgt]["target_normal_latent"],
                    seed=args.seed + pair_index,
                    max_samples=args.max_pad_samples,
                    logreg_c=args.pad_logreg_c,
                )
                row = {
                    "source_entity": src,
                    "target_entity": tgt,
                    "family": src_family,
                    "n_source_normal_windows": int(len(src_pack["source_normal_eval"])),
                    "n_target_normal_windows": int(len(tgt_pack["target_normal_eval"])),
                    "pad_latent": pad_metrics,
                }
                if args.include_summary_baseline:
                    row["summary_baseline"] = compute_domain_shift_metrics(
                        src_pack["source_normal_eval"],
                        tgt_pack["target_normal_eval"],
                        seed=args.seed + pair_index,
                        max_samples=args.max_pad_samples,
                    )
                rows.append(row)
                pair_index += 1
        rows.sort(key=lambda x: x["pad_latent"]["pad_value"], reverse=True)

    payload = {
        "method": "ben_david_plus_ts_jepa_pad_cached",
        "notes": {
            "primary_metric": "PAD_latent",
            "target_protocol": (
                "source normal vs target normal from the same cached entity"
                if args.pair_scope == "same_entity"
                else "source normal vs target normal across cached entities"
            ),
            "dataset_name": dataset_name,
            "pretrain_encoder": "global TS-JEPA-style encoder",
            "pair_scope": args.pair_scope,
        },
        "data_root": str(data_root),
        "device": device,
        "config": cfg.to_dict(),
        "max_train_windows_per_entity": args.max_train_windows_per_entity,
        "max_eval_windows_per_entity": args.max_eval_windows_per_entity,
        "max_pad_samples": args.max_pad_samples,
        "pad_logreg_c": args.pad_logreg_c,
        "entities": [p.name for p in entity_dirs],
        "train_history": train_history,
        "entity_stats": {
            entity_name: {
                "source_windows": int(len(pack["source_windows"])),
                "source_normal_windows": int(len(pack["source_normal_windows"])),
                "target_windows": int(len(pack["target_windows"])),
                "target_normal_windows": int(len(pack["target_normal_windows"])),
                "target_anomaly_windows": int((pack["target_labels"] == 1).sum()),
            }
            for entity_name, pack in entity_cache.items()
        },
        ("entity_rankings" if args.pair_scope == "same_entity" else "pair_rankings"): rows,
        ("top_entities" if args.pair_scope == "same_entity" else "top_pairs_by_source"): (
            top_entities(rows, args.topk)
            if args.pair_scope == "same_entity"
            else top_pairs_by_source(rows)
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "pilot_rankings.json"
    save_json(result_path, payload)
    print(f"[OK] Saved rankings: {result_path}")

    if args.pair_scope == "same_entity":
        print(f"\n[TOP {min(args.topk, len(rows))} PAD_latent entities]")
        for row in rows[: args.topk]:
            print(
                f"  {row['entity']} | "
                f"PAD={row['pad_latent']['pad_value']:.4f} | "
                f"ACC={row['pad_latent']['domain_acc']:.4f} | "
                f"AUC={row['pad_latent']['domain_auc']:.4f}"
            )
    else:
        print(f"\n[TOP {min(args.topk, len(rows))} PAD_latent pairs]")
        for row in rows[: args.topk]:
            print(
                f"  {row['source_entity']} -> {row['target_entity']} | "
                f"PAD={row['pad_latent']['pad_value']:.4f} | "
                f"ACC={row['pad_latent']['domain_acc']:.4f} | "
                f"AUC={row['pad_latent']['domain_auc']:.4f}"
            )


if __name__ == "__main__":
    main()
