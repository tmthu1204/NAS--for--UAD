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
    ap.add_argument("--raw_smd_root", default=os.path.join(PROJ, "data", "ServerMachineDataset"))
    ap.add_argument("--machine", default="machine-1-1")
    ap.add_argument("--search_candidates", default="4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tranad_epochs", default="5")
    ap.add_argument("--tranad_final_epochs", default="5")
    ap.add_argument("--tranad_lr", default="0.0001")
    ap.add_argument("--tranad_patience", default="5")
    ap.add_argument("--tranad_window_length", default="10")
    ap.add_argument("--tranad_valid_ratio", default="0.2")
    ap.add_argument("--tranad_batch_size", default="128")
    ap.add_argument("--tranad_ff_dim", default="0")
    ap.add_argument("--tranad_dropout", default="0.1")
    ap.add_argument("--tranad_encoder_layers", default="0")
    ap.add_argument("--tranad_decoder_layers", default="0")
    ap.add_argument("--tranad_search_iters", default="2")
    ap.add_argument("--tranad_train_start", default="0")
    ap.add_argument("--tranad_train_limit", default="0")
    ap.add_argument("--tranad_test_start", default="0")
    ap.add_argument("--tranad_test_limit", default="0")
    ap.add_argument("--tranad_pot_q", default="0.001")
    ap.add_argument("--tranad_pot_level", default="0.99")
    ap.add_argument("--tranad_fixed_only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    py = sys.executable
    bench_name = "tranad-uad_source"
    if args.tag.strip():
        bench_name = f"{bench_name}-{args.tag.strip()}"
    bench_dir = os.path.join(BENCHMARKS, bench_name)
    os.makedirs(bench_dir, exist_ok=True)

    log_file = os.path.join(LOGS, f"{bench_name}-{args.machine}.txt")
    cmd = [
        py, "-m", "src.pipeline",
        "--mode", "uad_source",
        "--family", "tranad",
        "--raw_smd_root", os.path.relpath(args.raw_smd_root, PROJ) if os.path.isabs(args.raw_smd_root) else args.raw_smd_root,
        "--machine", args.machine,
        "--search_candidates", args.search_candidates,
        "--device", args.device,
        "--tranad_epochs", args.tranad_epochs,
        "--tranad_final_epochs", args.tranad_final_epochs,
        "--tranad_lr", args.tranad_lr,
        "--tranad_patience", args.tranad_patience,
        "--tranad_window_length", args.tranad_window_length,
        "--tranad_valid_ratio", args.tranad_valid_ratio,
        "--tranad_batch_size", args.tranad_batch_size,
        "--tranad_ff_dim", args.tranad_ff_dim,
        "--tranad_dropout", args.tranad_dropout,
        "--tranad_encoder_layers", args.tranad_encoder_layers,
        "--tranad_decoder_layers", args.tranad_decoder_layers,
        "--tranad_search_iters", args.tranad_search_iters,
        "--tranad_train_start", args.tranad_train_start,
        "--tranad_train_limit", args.tranad_train_limit,
        "--tranad_test_start", args.tranad_test_start,
        "--tranad_test_limit", args.tranad_test_limit,
        "--tranad_pot_q", args.tranad_pot_q,
        "--tranad_pot_level", args.tranad_pot_level,
    ]
    if args.tranad_fixed_only:
        cmd.append("--tranad_fixed_only")

    rc = run_cmd(cmd, log_file)
    if rc != 0:
        print(f"[FAIL] TranAD run failed (code {rc}). See log: {os.path.relpath(log_file, PROJ)}")
        sys.exit(rc)

    result = load_results_json()
    if result is None:
        print("[FAIL] outputs/results.json not found after run.")
        sys.exit(1)

    save_json(os.path.join(bench_dir, f"{args.machine}.json"), result)
    print(f"[OK] Saved benchmark result to {os.path.relpath(os.path.join(bench_dir, f'{args.machine}.json'), PROJ)}")


if __name__ == "__main__":
    main()
