import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.make_uad_smd import (
    binarize_y,
    compute_domain_shift_metrics,
    create_dataset,
    load_npz,
)


def list_channel_dirs(data_root: Path):
    return sorted([p for p in data_root.iterdir() if p.is_dir() and (p / "source.npz").exists() and (p / "target.npz").exists()])


def channel_family(name: str) -> str:
    return name.split("-")[0] if "-" in name else name


def source_norm_windows(channel_dir: Path):
    Xs, ys = load_npz(str(channel_dir / "source.npz"))
    ys = binarize_y(ys)
    return Xs[ys == 0]


def target_windows(channel_dir: Path):
    Xt, yt = load_npz(str(channel_dir / "target.npz"))
    yt = binarize_y(yt)
    return Xt, yt


def candidate_cross_targets(source_dir: Path, all_dirs, same_prefix_only: bool):
    src_family = channel_family(source_dir.name)
    out = []
    for target_dir in all_dirs:
        if target_dir == source_dir:
            continue
        if same_prefix_only and channel_family(target_dir.name) != src_family:
            continue
        out.append(target_dir)
    return out


def rank_cross_targets(source_dir: Path, target_dirs, min_target_anom: int):
    Xs_norm = source_norm_windows(source_dir)
    ranked = []
    for target_dir in target_dirs:
        Xt, yt = target_windows(target_dir)
        target_anom = int((yt == 1).sum())
        if target_anom < min_target_anom:
            continue
        shift = compute_domain_shift_metrics(Xs_norm, Xt, seed=42)
        auc = shift["domain_auc"]
        if not np.isfinite(auc):
            auc = 0.5
        ranked.append((float(auc), target_anom, target_dir, shift))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def build_args(
    *,
    source_dir: Path,
    target_dir: Path | None,
    out_dir: Path,
    split_mode: str,
    shift_level: str,
    target_pool_frac: float,
    val_frac: float,
    guard: int,
    search_step: int,
    max_pool_anom_ratio: float,
    min_target_pool: int,
    min_val: int,
    min_test: int,
    min_anom_val: int,
    min_anom_test: int,
    seed: int,
):
    return SimpleNamespace(
        machine_dir=str(source_dir),
        target_machine_dir=(str(target_dir) if target_dir is not None else None),
        source_name="source.npz",
        target_name="target.npz",
        out_dir=str(out_dir),
        out_train="train_normal.npz",
        out_target_pool="target_pool_unlabeled.npz",
        out_val="val_mixed.npz",
        out_test="test_mixed.npz",
        out_meta="split_metadata.json",
        split_mode=split_mode,
        shift_level=shift_level,
        train_normal_frac=1.0,
        target_pool_frac=target_pool_frac,
        val_frac=val_frac,
        guard=guard,
        search_step=search_step,
        max_pool_anom_ratio=max_pool_anom_ratio,
        min_train=0,
        min_target_pool=min_target_pool,
        min_val=min_val,
        min_test=min_test,
        min_anom_val=min_anom_val,
        min_anom_test=min_anom_test,
        allow_single_class_eval=False,
        seed=seed,
        strict=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/msl")
    ap.add_argument("--out_root", default="data/msl_experiments")
    ap.add_argument("--channels", default=None, help="Comma-separated channel ids to include, e.g. M-1,M-2")
    ap.add_argument("--max_channels", type=int, default=0)
    ap.add_argument("--shift_levels", default="auto,hard")
    ap.add_argument("--build_temporal", action="store_true")
    ap.add_argument("--build_cross_entity", action="store_true")
    ap.add_argument(
        "--allow_cross_prefix",
        action="store_true",
        help="If set, allow cross-prefix target channels. By default, MSL experiments stay within the same channel prefix family.",
    )
    ap.add_argument("--topk_cross", type=int, default=1)
    ap.add_argument("--target_pool_frac", type=float, default=0.10)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--guard", type=int, default=0)
    ap.add_argument("--search_step", type=int, default=2)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=0.15)
    ap.add_argument("--min_target_pool", type=int, default=8)
    ap.add_argument("--min_val", type=int, default=8)
    ap.add_argument("--min_test", type=int, default=12)
    ap.add_argument("--min_anom_val", type=int, default=1)
    ap.add_argument("--min_anom_test", type=int, default=1)
    ap.add_argument("--min_target_anom", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    shift_levels = [s.strip() for s in args.shift_levels.split(",") if s.strip()]
    channels = list_channel_dirs(data_root)
    if not channels:
        raise FileNotFoundError(f"No channel folders with source.npz/target.npz under {data_root}")

    if args.channels:
        keep = {c.strip() for c in args.channels.split(",") if c.strip()}
        channels = [c for c in channels if c.name in keep]
    if args.max_channels > 0:
        channels = channels[: args.max_channels]
    if not channels:
        raise ValueError("No channels left after filtering.")

    build_temporal = args.build_temporal or (not args.build_temporal and not args.build_cross_entity)
    build_cross = args.build_cross_entity or (not args.build_temporal and not args.build_cross_entity)
    same_prefix_only = not args.allow_cross_prefix

    manifest = []

    if build_temporal:
        for shift_level in shift_levels:
            for source_dir in channels:
                out_dir = out_root / f"temporal_{shift_level}" / source_dir.name
                ds_args = build_args(
                    source_dir=source_dir,
                    target_dir=None,
                    out_dir=out_dir,
                    split_mode="search",
                    shift_level=shift_level,
                    target_pool_frac=args.target_pool_frac,
                    val_frac=args.val_frac,
                    guard=args.guard,
                    search_step=args.search_step,
                    max_pool_anom_ratio=args.max_pool_anom_ratio,
                    min_target_pool=args.min_target_pool,
                    min_val=args.min_val,
                    min_test=args.min_test,
                    min_anom_val=args.min_anom_val,
                    min_anom_test=args.min_anom_test,
                    seed=args.seed,
                )
                try:
                    meta = create_dataset(ds_args)
                    manifest.append(meta)
                except Exception as exc:
                    print(f"[WARN] temporal {shift_level} {source_dir.name}: {exc}")

    if build_cross:
        for shift_level in shift_levels:
            for source_dir in channels:
                ranked = rank_cross_targets(
                    source_dir,
                    candidate_cross_targets(source_dir, channels, same_prefix_only),
                    min_target_anom=args.min_target_anom,
                )
                for _, _, target_dir, shift in ranked[: args.topk_cross]:
                    out_dir = out_root / f"cross_entity_{shift_level}" / f"{source_dir.name}__to__{target_dir.name}"
                    ds_args = build_args(
                        source_dir=source_dir,
                        target_dir=target_dir,
                        out_dir=out_dir,
                        split_mode="search",
                        shift_level=shift_level,
                        target_pool_frac=args.target_pool_frac,
                        val_frac=args.val_frac,
                        guard=args.guard,
                        search_step=args.search_step,
                        max_pool_anom_ratio=args.max_pool_anom_ratio,
                        min_target_pool=args.min_target_pool,
                        min_val=args.min_val,
                        min_test=args.min_test,
                        min_anom_val=args.min_anom_val,
                        min_anom_test=args.min_anom_test,
                        seed=args.seed,
                    )
                    try:
                        meta = create_dataset(ds_args)
                        meta["candidate_pair_shift_precheck"] = shift
                        meta["same_prefix_only"] = bool(same_prefix_only)
                        manifest.append(meta)
                    except Exception as exc:
                        print(f"[WARN] cross {shift_level} {source_dir.name}->{target_dir.name}: {exc}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Total experiment folders: {len(manifest)}")


if __name__ == "__main__":
    main()
