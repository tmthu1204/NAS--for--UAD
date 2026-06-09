import argparse
from pathlib import Path
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.preprocess_smap import run_preprocess
from src.utils.data_paths import resolve_raw_smap_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_root",
        type=str,
        default="data/SMAP_MSL",
        help="Directory containing train/, test/, and labeled_anomalies.csv for Telemanom MSL.",
    )
    ap.add_argument("--out_root", type=str, default="data/msl")
    ap.add_argument("--channel", type=str, default=None, help="Example: M-6. If omitted, preprocess all MSL channels.")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    args = ap.parse_args()

    raw_root = resolve_raw_smap_root(args.raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw Telemanom root not found: {raw_root}")

    run_preprocess(
        raw_root=raw_root,
        out_root=Path(args.out_root),
        spacecraft="MSL",
        channel=args.channel,
        window=args.window,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
