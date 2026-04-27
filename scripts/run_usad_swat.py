import argparse
import json
import math
import os
import subprocess
import sys

import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
LOGS = os.path.join(PROJ, "outputs", "logs")
BENCHMARKS = os.path.join(PROJ, "outputs", "benchmarks")
os.makedirs(LOGS, exist_ok=True)
os.makedirs(BENCHMARKS, exist_ok=True)


def run_cmd(args, log_path):
    print(">>", " ".join(args))
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, cwd=PROJ)
        proc.wait()
        return proc.returncode


def load_results_json():
    path = os.path.join(PROJ, "outputs", "results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=os.path.join(PROJ, "data", "SWaT", "SWaT_Dataset_Normal_v1.csv"))
    ap.add_argument("--test_csv", default=os.path.join(PROJ, "data", "SWaT", "SWaT_Dataset_Attack_v0.csv"))
    ap.add_argument("--search_candidates", default="5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--usad_epochs", default="70")
    ap.add_argument("--usad_final_epochs", default="70")
    ap.add_argument("--usad_final_patience", default="0")
    ap.add_argument("--usad_lr", default="0.001")
    ap.add_argument("--usad_patience", default="5")
    ap.add_argument("--usad_window_length", default="12")
    ap.add_argument("--usad_valid_ratio", default="0.2")
    ap.add_argument("--usad_batch_size", default="128")
    ap.add_argument("--usad_stride", default="1")
    ap.add_argument("--usad_downsample", default="5")
    ap.add_argument("--usad_latent_size", default="0")
    ap.add_argument("--usad_search_iters", default="3")
    ap.add_argument("--usad_train_limit", default="0")
    ap.add_argument("--usad_test_limit", default="0")
    ap.add_argument("--usad_score_alpha", default="0.5")
    ap.add_argument("--usad_score_beta", default="0.5")
    ap.add_argument("--usad_preprocess", default="train_minmax", choices=["train_minmax", "train_zscore"])
    ap.add_argument("--usad_pot_q", default="0.001")
    ap.add_argument("--usad_pot_level", default="0.99")
    ap.add_argument("--usad_fixed_only", action="store_true")
    ap.add_argument("--usad_final_early_stopping", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    py = sys.executable
    bench_name = "usad-uad_source"
    if args.tag.strip():
        bench_name = f"{bench_name}-{args.tag.strip()}"
    bench_dir = os.path.join(BENCHMARKS, bench_name)
    os.makedirs(bench_dir, exist_ok=True)

    log_file = os.path.join(LOGS, f"{bench_name}-swat.txt")
    cmd = [
        py, "-m", "src.pipeline",
        "--mode", "uad_source",
        "--family", "usad",
        "--swat_train_csv", os.path.relpath(args.train_csv, PROJ) if os.path.isabs(args.train_csv) else args.train_csv,
        "--swat_test_csv", os.path.relpath(args.test_csv, PROJ) if os.path.isabs(args.test_csv) else args.test_csv,
        "--search_candidates", args.search_candidates,
        "--device", args.device,
        "--usad_epochs", args.usad_epochs,
        "--usad_final_epochs", args.usad_final_epochs,
        "--usad_final_patience", args.usad_final_patience,
        "--usad_lr", args.usad_lr,
        "--usad_patience", args.usad_patience,
        "--usad_window_length", args.usad_window_length,
        "--usad_valid_ratio", args.usad_valid_ratio,
        "--usad_batch_size", args.usad_batch_size,
        "--usad_stride", args.usad_stride,
        "--usad_downsample", args.usad_downsample,
        "--usad_latent_size", args.usad_latent_size,
        "--usad_search_iters", args.usad_search_iters,
        "--usad_train_limit", args.usad_train_limit,
        "--usad_test_limit", args.usad_test_limit,
        "--usad_score_alpha", args.usad_score_alpha,
        "--usad_score_beta", args.usad_score_beta,
        "--usad_preprocess", args.usad_preprocess,
        "--usad_pot_q", args.usad_pot_q,
        "--usad_pot_level", args.usad_pot_level,
    ]
    if args.usad_fixed_only:
        cmd.append("--usad_fixed_only")
    if args.usad_final_early_stopping:
        cmd.append("--usad_final_early_stopping")

    rc = run_cmd(cmd, log_file)
    if rc != 0:
        print(f"[FAIL] SWaT run failed (code {rc}). See log: {os.path.relpath(log_file, PROJ)}")
        sys.exit(rc)

    result = load_results_json()
    if result is None:
        print("[FAIL] outputs/results.json not found after run.")
        sys.exit(1)

    save_json(os.path.join(bench_dir, "swat.json"), result)
    print(f"[OK] Saved benchmark result to {os.path.relpath(os.path.join(bench_dir, 'swat.json'), PROJ)}")


if __name__ == "__main__":
    main()
