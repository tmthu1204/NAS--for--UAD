import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.make_uad_smd import create_dataset


def infer_default_out_dir(source_dir: Path, target_dir: Path, shift_level: str) -> Path:
    same_entity = source_dir.resolve() == target_dir.resolve()
    root = source_dir.parent.parent / "msl_experiments"
    if same_entity:
        return root / f"temporal_{shift_level}" / source_dir.name
    return root / f"cross_entity_{shift_level}" / f"{source_dir.name}__to__{target_dir.name}"


def build_args(parsed):
    source_dir = Path(parsed.channel_dir)
    target_dir = Path(parsed.target_channel_dir) if parsed.target_channel_dir else source_dir
    out_dir = Path(parsed.out_dir) if parsed.out_dir else infer_default_out_dir(source_dir, target_dir, parsed.shift_level)

    return SimpleNamespace(
        machine_dir=str(source_dir),
        target_machine_dir=str(target_dir) if parsed.target_channel_dir else None,
        source_name="source.npz",
        target_name="target.npz",
        out_dir=str(out_dir),
        out_train="train_normal.npz",
        out_target_pool="target_pool_unlabeled.npz",
        out_val="val_mixed.npz",
        out_test="test_mixed.npz",
        out_meta="split_metadata.json",
        split_mode=parsed.split_mode,
        shift_level=parsed.shift_level,
        train_normal_frac=parsed.train_normal_frac,
        target_pool_frac=parsed.target_pool_frac,
        val_frac=parsed.val_frac,
        guard=parsed.guard,
        search_step=parsed.search_step,
        max_pool_anom_ratio=parsed.max_pool_anom_ratio,
        min_train=parsed.min_train,
        min_target_pool=parsed.min_target_pool,
        min_val=parsed.min_val,
        min_test=parsed.min_test,
        min_anom_val=parsed.min_anom_val,
        min_anom_test=parsed.min_anom_test,
        allow_single_class_eval=parsed.allow_single_class_eval,
        seed=parsed.seed,
        strict=parsed.strict,
    )


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--channel_dir",
        required=True,
        help="Source MSL channel directory, e.g. data/msl/M-1",
    )
    ap.add_argument(
        "--target_channel_dir",
        default=None,
        help="Optional target MSL channel directory for cross-entity covariate shift. If omitted, use the same channel.",
    )
    ap.add_argument("--out_dir", default=None)

    ap.add_argument("--split_mode", choices=["search", "fixed"], default="search")
    ap.add_argument("--shift_level", choices=["auto", "hard", "medium", "mild"], default="auto")
    ap.add_argument("--train_normal_frac", type=float, default=1.0)
    ap.add_argument("--target_pool_frac", type=float, default=0.10)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--guard", type=int, default=0)
    ap.add_argument("--search_step", type=int, default=2)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=0.15)

    ap.add_argument("--min_train", type=int, default=0)
    ap.add_argument("--min_target_pool", type=int, default=8)
    ap.add_argument("--min_val", type=int, default=8)
    ap.add_argument("--min_test", type=int, default=12)
    ap.add_argument("--min_anom_val", type=int, default=1)
    ap.add_argument("--min_anom_test", type=int, default=1)
    ap.add_argument("--allow_single_class_eval", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true")
    return ap


def main():
    parsed = build_arg_parser().parse_args()
    create_dataset(build_args(parsed))


if __name__ == "__main__":
    main()
