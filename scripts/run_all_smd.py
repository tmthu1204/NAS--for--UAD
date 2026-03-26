import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import math


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
LOGS = os.path.join(PROJ, "outputs", "logs")
os.makedirs(LOGS, exist_ok=True)
BENCHMARKS = os.path.join(PROJ, "outputs", "benchmarks")
os.makedirs(BENCHMARKS, exist_ok=True)


def have(*paths):
    return all(os.path.exists(p) for p in paths)


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


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)


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


def mean_of_metric(entries, section, metric):
    vals = []
    for item in entries:
        val = item.get(section, {}).get(metric)
        if isinstance(val, (int, float)):
            vals.append(float(val))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_omni_summary(entries):
    metric_keys = [
        "ap",
        "auroc",
        "f1_best",
        "f1_pot",
        "event_f1",
        "delay_mean",
    ]
    summary_rows = []
    for item in entries:
        fixed = item["fixed"]
        searched = item["searched"]
        delta = {}
        for key in metric_keys:
            f = fixed.get(key)
            s = searched.get(key)
            if isinstance(f, (int, float)) and isinstance(s, (int, float)):
                delta[key] = float(s - f)
        summary_rows.append(
            {
                "machine": item["machine"],
                "fixed_baseline": fixed,
                "searched_partial_nas": searched,
                "delta_searched_minus_fixed": delta,
            }
        )

    macro = {
        "num_machines": len(summary_rows),
        "fixed_baseline_mean": {
            key: mean_of_metric(summary_rows, "fixed_baseline", key)
            for key in metric_keys
        },
        "searched_partial_nas_mean": {
            key: mean_of_metric(summary_rows, "searched_partial_nas", key)
            for key in metric_keys
        },
        "delta_mean": {
            key: mean_of_metric(summary_rows, "delta_searched_minus_fixed", key)
            for key in metric_keys
        },
    }
    return {
        "rows": summary_rows,
        "macro": macro,
    }


def collect_omni_entries_from_dir(bench_dir):
    entries = []
    for path in sorted(Path(bench_dir).glob("machine-*.json")):
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        searched_block = result.get("searched_partial_nas") or {}
        entries.append(
            {
                "machine": result.get("machine", path.stem),
                "fixed": result.get("fixed_baseline", {}).get("metrics_uad", {}),
                "searched": searched_block.get("metrics_uad", {}),
            }
        )
    return entries


def build_dataset_arg(machine_dir, mode):
    train_n = os.path.join(machine_dir, "train_normal.npz")
    target_pool = os.path.join(machine_dir, "target_pool_unlabeled.npz")
    val_m = os.path.join(machine_dir, "val_mixed.npz")
    test_m = os.path.join(machine_dir, "test_mixed.npz")

    if mode == "uad_source":
        if not have(train_n, val_m):
            return None
        parts = [train_n, val_m]
        if os.path.exists(test_m):
            parts.append(test_m)
        return ",".join(os.path.relpath(p, PROJ) for p in parts)

    if not have(train_n, val_m):
        return None

    if os.path.exists(target_pool):
        parts = [train_n, target_pool, val_m]
        if os.path.exists(test_m):
            parts.append(test_m)
        return ",".join(os.path.relpath(p, PROJ) for p in parts)

    if os.path.exists(test_m):
        print(f"[WARN] {os.path.basename(machine_dir)} has no target_pool_unlabeled.npz; using legacy 3-file fallback.")
        parts = [train_n, val_m, test_m]
        return ",".join(os.path.relpath(p, PROJ) for p in parts)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="uad_source", choices=["uad_source", "adaptnas_combined"])
    ap.add_argument("--family", default="default_nasade", choices=["default_nasade", "omni_anomaly"])
    ap.add_argument("--epochs_pretrain", default="5")
    ap.add_argument("--search_candidates", default="5")
    ap.add_argument("--batch_size", default="64")
    ap.add_argument("--data_root", default=os.path.join(PROJ, "data", "smd"))
    ap.add_argument("--raw_smd_root", default=os.path.join(PROJ, "data", "ServerMachineDataset"))
    ap.add_argument("--machines", default="", help="Comma-separated machine ids for omni_anomaly, e.g. machine-1-1,machine-1-2")
    ap.add_argument("--omni_epochs", default="20")
    ap.add_argument("--omni_final_epochs", default="20")
    ap.add_argument("--omni_lr", default="0.001")
    ap.add_argument("--omni_patience", default="5")
    ap.add_argument("--omni_window_length", default="100")
    ap.add_argument("--omni_valid_ratio", default="0.3")
    ap.add_argument("--omni_batch_size", default="50")
    ap.add_argument("--omni_stride", default="1")
    ap.add_argument("--omni_test_n_z", default="1")
    ap.add_argument("--omni_search_iters", default="3")
    ap.add_argument("--omni_train_limit", default="0")
    ap.add_argument("--omni_test_limit", default="0")
    ap.add_argument("--omni_reference", default="paper", choices=["paper", "repo"])
    ap.add_argument("--omni_preprocess", default="official_minmax", choices=["official_minmax", "train_zscore"])
    ap.add_argument("--omni_fixed_only", action="store_true")
    ap.add_argument("--omni_pot_q", default="0")
    ap.add_argument("--omni_pot_level", default="0")
    ap.add_argument("--tag", default="", help="Optional suffix for benchmark output folder.")
    args = ap.parse_args()

    py = sys.executable
    ok, fail = [], []
    bench_name = f"{args.family}-{args.mode}"
    if args.tag.strip():
        bench_name = f"{bench_name}-{args.tag.strip()}"
    bench_dir = os.path.join(BENCHMARKS, bench_name)
    os.makedirs(bench_dir, exist_ok=True)
    omni_summary_entries = []

    if args.family == "omni_anomaly":
        if args.mode != "uad_source":
            print("family=omni_anomaly currently supports only mode=uad_source")
            sys.exit(1)

        raw_root = Path(args.raw_smd_root)
        if not raw_root.exists():
            print(f"Raw SMD root not found: {raw_root}")
            sys.exit(1)

        if args.machines.strip():
            machines = [m.strip() for m in args.machines.split(",") if m.strip()]
        else:
            machines = sorted(p.stem for p in (raw_root / "train").glob("*.txt"))

        if not machines:
            print(f"No machine-*.txt files under {raw_root / 'train'}")
            sys.exit(1)

        for machine in machines:
            print(f"\n==== RUN {machine} ====")
            log_file = os.path.join(LOGS, f"{args.family}-{args.mode}-{machine}.txt")
            cmd = [
                py, "-m", "src.pipeline",
                "--mode", args.mode,
                "--family", args.family,
                "--raw_smd_root", os.path.relpath(str(raw_root), PROJ),
                "--machine", machine,
                "--epochs_pretrain", args.epochs_pretrain,
                "--search_candidates", args.search_candidates,
                "--batch_size", args.batch_size,
                "--omni_epochs", args.omni_epochs,
                "--omni_final_epochs", args.omni_final_epochs,
                "--omni_lr", args.omni_lr,
                "--omni_patience", args.omni_patience,
                "--omni_window_length", args.omni_window_length,
                "--omni_valid_ratio", args.omni_valid_ratio,
                "--omni_batch_size", args.omni_batch_size,
                "--omni_stride", args.omni_stride,
                "--omni_test_n_z", args.omni_test_n_z,
                "--omni_search_iters", args.omni_search_iters,
                "--omni_train_limit", args.omni_train_limit,
                "--omni_test_limit", args.omni_test_limit,
                "--omni_reference", args.omni_reference,
                "--omni_preprocess", args.omni_preprocess,
                "--omni_pot_q", args.omni_pot_q,
                "--omni_pot_level", args.omni_pot_level,
            ]
            if args.omni_fixed_only:
                cmd.append("--omni_fixed_only")
            rc = run_cmd(cmd, log_file)
            if rc == 0:
                result = load_results_json()
                if result is not None:
                    save_json(os.path.join(bench_dir, f"{machine}.json"), result)
                    fixed = result.get("fixed_baseline", {}).get("metrics_uad", {})
                    searched_block = result.get("searched_partial_nas") or {}
                    searched = searched_block.get("metrics_uad", {})
                    omni_summary_entries.append(
                        {
                            "machine": machine,
                            "fixed": fixed,
                            "searched": searched,
                        }
                    )
                ok.append(machine)
            else:
                print(f"[FAIL] {machine} (code {rc}). See log: {os.path.relpath(log_file, PROJ)}")
                fail.append(machine)
    else:
        data_root = args.data_root
        machine_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        machine_dirs.sort()
        if not machine_dirs:
            print(f"No subfolders under {data_root}")
            sys.exit(1)

        for machine_dir in machine_dirs:
            machine = os.path.basename(machine_dir)
            print(f"\n==== RUN {machine} ====")
            ds_arg = build_dataset_arg(machine_dir, args.mode)
            if ds_arg is None:
                print(f"[SKIP] Missing required files in {machine_dir}")
                fail.append(machine)
                continue

            log_file = os.path.join(LOGS, f"{args.family}-{args.mode}-{machine}.txt")
            cmd = [
                py, "-m", "src.pipeline",
                "--dataset_or_paths", ds_arg,
                "--mode", args.mode,
                "--family", args.family,
                "--epochs_pretrain", args.epochs_pretrain,
                "--search_candidates", args.search_candidates,
                "--batch_size", args.batch_size,
            ]

            rc = run_cmd(cmd, log_file)
            if rc == 0:
                result = load_results_json()
                if result is not None:
                    save_json(os.path.join(bench_dir, f"{machine}.json"), result)
                ok.append(machine)
            else:
                print(f"[FAIL] {machine} (code {rc}). See log: {os.path.relpath(log_file, PROJ)}")
                fail.append(machine)

    if args.family == "omni_anomaly":
        all_entries = collect_omni_entries_from_dir(bench_dir)
        if all_entries:
            save_json(
                os.path.join(bench_dir, "summary.json"),
                build_omni_summary(all_entries),
            )

    print("\n==== SUMMARY ====")
    print("OK  :", ok)
    print("FAIL:", fail)


if __name__ == "__main__":
    main()
