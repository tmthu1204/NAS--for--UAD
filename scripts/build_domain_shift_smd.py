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


def list_machine_dirs(data_root: Path):
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("machine-")])


def machine_family(name: str) -> str:
    parts = name.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else name


def source_norm_windows(machine_dir: Path):
    Xs, ys = load_npz(str(machine_dir / "source.npz"))
    ys = binarize_y(ys)
    return Xs[ys == 0]


def target_windows(machine_dir: Path):
    Xt, yt = load_npz(str(machine_dir / "target.npz"))
    yt = binarize_y(yt)
    return Xt, yt


def candidate_cross_targets(source_dir: Path, all_dirs, same_family_only: bool):
    src_family = machine_family(source_dir.name)
    out = []
    for target_dir in all_dirs:
        if target_dir == source_dir:
            continue
        if same_family_only and machine_family(target_dir.name) != src_family:
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
    ap.add_argument("--data_root", default="data/smd")
    ap.add_argument("--out_root", default="data/smd_experiments")
    ap.add_argument("--machines", default=None, help="Comma-separated machine names to include, e.g. machine-1-1,machine-1-2")
    ap.add_argument("--max_machines", type=int, default=0)
    ap.add_argument("--shift_levels", default="medium,hard")
    ap.add_argument("--build_temporal", action="store_true")
    ap.add_argument("--build_cross_machine", action="store_true")
    ap.add_argument("--same_family_only", action="store_true")
    ap.add_argument("--topk_cross", type=int, default=1)
    ap.add_argument("--target_pool_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.30)
    ap.add_argument("--guard", type=int, default=4)
    ap.add_argument("--search_step", type=int, default=4)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=0.10)
    ap.add_argument("--min_target_pool", type=int, default=32)
    ap.add_argument("--min_val", type=int, default=32)
    ap.add_argument("--min_test", type=int, default=64)
    ap.add_argument("--min_anom_val", type=int, default=3)
    ap.add_argument("--min_anom_test", type=int, default=5)
    ap.add_argument("--min_target_anom", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    shift_levels = [s.strip() for s in args.shift_levels.split(",") if s.strip()]
    machines = list_machine_dirs(data_root)
    if not machines:
        raise FileNotFoundError(f"No machine-* folders under {data_root}")

    if args.machines:
        keep = {m.strip() for m in args.machines.split(",") if m.strip()}
        machines = [m for m in machines if m.name in keep]
    if args.max_machines > 0:
        machines = machines[: args.max_machines]
    if not machines:
        raise ValueError("No machines left after filtering.")

    build_temporal = args.build_temporal or (not args.build_temporal and not args.build_cross_machine)
    build_cross = args.build_cross_machine or (not args.build_temporal and not args.build_cross_machine)

    manifest = []

    if build_temporal:
        for shift_level in shift_levels:
            for source_dir in machines:
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
            for source_dir in machines:
                ranked = rank_cross_targets(
                    source_dir,
                    candidate_cross_targets(source_dir, machines, args.same_family_only),
                    min_target_anom=args.min_target_anom,
                )
                for _, _, target_dir, shift in ranked[: args.topk_cross]:
                    out_dir = out_root / f"cross_machine_{shift_level}" / f"{source_dir.name}__to__{target_dir.name}"
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
