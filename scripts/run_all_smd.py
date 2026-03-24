import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
LOGS = os.path.join(PROJ, "outputs", "logs")
os.makedirs(LOGS, exist_ok=True)


def have(*paths):
    return all(os.path.exists(p) for p in paths)


def run_cmd(args, log_path):
    print(">>", " ".join(args))
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, cwd=PROJ)
        proc.wait()
        return proc.returncode


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
    ap.add_argument("--epochs_pretrain", default="5")
    ap.add_argument("--search_candidates", default="5")
    ap.add_argument("--batch_size", default="64")
    ap.add_argument("--data_root", default=os.path.join(PROJ, "data", "smd"))
    args = ap.parse_args()

    data_root = args.data_root
    machine_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    machine_dirs.sort()
    if not machine_dirs:
        print(f"No subfolders under {data_root}")
        sys.exit(1)

    py = sys.executable
    ok, fail = [], []

    for machine_dir in machine_dirs:
        machine = os.path.basename(machine_dir)
        print(f"\n==== RUN {machine} ====")
        ds_arg = build_dataset_arg(machine_dir, args.mode)
        if ds_arg is None:
            print(f"[SKIP] Missing required files in {machine_dir}")
            fail.append(machine)
            continue

        log_file = os.path.join(LOGS, f"{args.mode}-{machine}.txt")
        cmd = [
            py, "-m", "src.pipeline",
            "--dataset_or_paths", ds_arg,
            "--mode", args.mode,
            "--epochs_pretrain", args.epochs_pretrain,
            "--search_candidates", args.search_candidates,
            "--batch_size", args.batch_size,
        ]

        rc = run_cmd(cmd, log_file)
        if rc == 0:
            ok.append(machine)
        else:
            print(f"[FAIL] {machine} (code {rc}). See log: {os.path.relpath(log_file, PROJ)}")
            fail.append(machine)

    print("\n==== SUMMARY ====")
    print("OK  :", ok)
    print("FAIL:", fail)


if __name__ == "__main__":
    main()
