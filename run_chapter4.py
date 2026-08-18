from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("smd", "msl", "smap")
REQUIRED_SPLIT_FILES = (
    "split_metadata.json",
    "train_normal.npz",
    "target_pool_unlabeled.npz",
    "val_mixed.npz",
    "test_mixed.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NAS-ADE experiments reported in thesis Chapter 4."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset to run; default: all three Chapter 4 datasets.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate Python, dependencies, manifests, and all packaged data files.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one pair with one-epoch settings as a smoke test, not thesis results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cases even when their output files already exist.",
    )
    return parser.parse_args()


def selected_datasets(name: str) -> tuple[str, ...]:
    return DATASETS if name == "all" else (name,)


def load_and_validate_manifest(dataset: str) -> list[dict]:
    manifest_path = ROOT / "data" / "chapter4" / dataset / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest is empty or invalid: {manifest_path}")

    for entry in entries:
        pair = f"{entry['source_entity']}__to__{entry['target_entity']}"
        split_dir = ROOT / entry["out_dir"]
        for file_name in REQUIRED_SPLIT_FILES:
            path = split_dir / file_name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing data for {dataset}/{pair}: {path}")
    return entries


def check_environment(datasets: tuple[str, ...]) -> object:
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported; "
            "use Python 3.12 as described in HuongDanCaiDat.txt."
        )

    try:
        import einops  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy as np
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "A required package is missing. Activate .venv and run "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    counts = {}
    for dataset in datasets:
        entries = load_and_validate_manifest(dataset)
        counts[dataset] = len(entries)
        # Opening every archive catches truncated files before a long experiment.
        for entry in entries:
            split_dir = ROOT / entry["out_dir"]
            for file_name in REQUIRED_SPLIT_FILES[1:]:
                with np.load(split_dir / file_name, allow_pickle=False) as archive:
                    if not archive.files:
                        raise ValueError(f"Empty NumPy archive: {split_dir / file_name}")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Packaged pairs: " + ", ".join(f"{k.upper()}={v}" for k, v in counts.items()))
    print("Environment and Chapter 4 data check: OK")
    return torch


def resolve_device(requested: str, torch: object) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot access a CUDA GPU.")
    return requested


def run_dataset(
    dataset: str,
    entries: list[dict],
    device: str,
    quick: bool,
    force: bool,
) -> None:
    manifest_path = ROOT / "data" / "chapter4" / dataset / "manifest.json"
    temporary_dir = None
    if quick:
        temporary_dir = tempfile.TemporaryDirectory(prefix="nas_ade_chapter4_")
        manifest_path = Path(temporary_dir.name) / f"{dataset}_quick_manifest.json"
        manifest_path.write_text(
            json.dumps(entries[:1], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_manifest_benchmarks.py"),
        "--manifest",
        str(manifest_path),
        "--dataset_name",
        f"chapter4_{dataset}{'_quick' if quick else ''}",
        "--output_root",
        str(ROOT / "outputs" / "chapter4"),
        "--device",
        device,
        "--seed",
        "42",
        "--epochs_pretrain",
        "1" if quick else "10",
        "--search_candidates",
        "1" if quick else "5",
        "--nas_search_iters",
        "1" if quick else "3",
        "--nas_search_strategy",
        "evolutionary_guided",
        "--combined_nas_topk_rerank",
        "1" if quick else "5",
        "--batch_size",
        "64",
        "--oneclass_method",
        "deepsvdd",
        "--weighting_oneclass_method",
        "deepsvdd",
        "--final_oneclass_method",
        "deepsvdd",
        "--oneclass_epochs",
        "1" if quick else "10",
        "--oneclass_final_epochs",
        "1" if quick else "20",
        "--oneclass_batch_size",
        "1024",
        "--oneclass_max_fit",
        "5000",
    ]
    if force:
        command.append("--force_run")

    run_kind = "quick smoke test" if quick else "full Chapter 4"
    print(f"\nRunning {dataset.upper()} on {device} ({run_kind})", flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


def main() -> int:
    args = parse_args()
    datasets = selected_datasets(args.dataset)
    torch = check_environment(datasets)
    if args.check_only:
        return 0

    device = resolve_device(args.device, torch)
    if args.quick:
        print("Quick mode validates execution only; it does not reproduce thesis metrics.")
    for dataset in datasets:
        run_dataset(dataset, load_and_validate_manifest(dataset), device, args.quick, args.force)

    output_root = ROOT / "outputs" / "chapter4"
    print(f"\nCompleted. Results are in: {output_root}")
    print(f"Reference values are in: {ROOT / 'reference_results' / 'chapter4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
