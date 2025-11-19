import os, sys, subprocess, shlex

ROOT = os.path.dirname(os.path.abspath(__file__))  # .../scripts
PROJ = os.path.dirname(ROOT)                       # project root
DATA = os.path.join(PROJ, "data", "smd")
LOGS = os.path.join(PROJ, "outputs", "logs")
os.makedirs(LOGS, exist_ok=True)

# cấu hình chung
EPOCHS_PRETRAIN = "5"
SEARCH_CANDIDATES = "5"
BATCH_SIZE = "64"
USE_UAD = "--uad" in sys.argv  # chạy: pypython scripts/run_all_smd.py --uad

def have(*paths):
    return all(os.path.exists(p) for p in paths)

def run_cmd(args, log_path):
    print(">>", " ".join(args))
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, cwd=PROJ)
        proc.wait()
        return proc.returncode

def main():
    machines = [d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))]
    machines.sort()
    if not machines:
        print(f"No subfolders under {DATA}")
        sys.exit(1)

    py = sys.executable  # đúng env hiện tại
    ok, fail = [], []

    for m in machines:
        print(f"\n==== RUN {m} ====")
        mdir = os.path.join(DATA, m)

        if USE_UAD:
            train_n = os.path.join(mdir, "train_normal.npz")
            val_m   = os.path.join(mdir, "val_mixed.npz")
            test_m  = os.path.join(mdir, "test_mixed.npz")  # optional
            if have(train_n, val_m):
                ds_arg = f"{os.path.relpath(train_n, PROJ)},{os.path.relpath(val_m, PROJ)}"
                if os.path.exists(test_m):
                    ds_arg += f",{os.path.relpath(test_m, PROJ)}"
                args = [
                    py, "-m", "src.pipeline",
                    "--dataset_or_paths", ds_arg,
                    "--uad",
                    "--epochs_pretrain", EPOCHS_PRETRAIN,
                    "--search_candidates", SEARCH_CANDIDATES,
                    "--batch_size", BATCH_SIZE
                ]
                log_file = os.path.join(LOGS, f"uad-{m}.txt")
            else:
                print(f"[SKIP] Missing UAD files in {mdir}")
                fail.append(m)
                continue
        else:
            src = os.path.join(mdir, "source.npz")
            tgt = os.path.join(mdir, "target.npz")
            if have(src, tgt):
                ds_arg = f"{os.path.relpath(src, PROJ)},{os.path.relpath(tgt, PROJ)}"
                args = [
                    py, "-m", "src.pipeline",
                    "--dataset_or_paths", ds_arg,
                    "--epochs_pretrain", EPOCHS_PRETRAIN,
                    "--search_candidates", SEARCH_CANDIDATES,
                    "--batch_size", BATCH_SIZE
                ]
                log_file = os.path.join(LOGS, f"smd-{m}.txt")
            else:
                print(f"[SKIP] Missing source/target in {mdir}")
                fail.append(m)
                continue

        rc = run_cmd(args, log_file)
        if rc == 0:
            ok.append(m)
        else:
            print(f"[FAIL] {m} (code {rc}). See log: {os.path.relpath(log_file, PROJ)}")
            fail.append(m)

    print("\n==== SUMMARY ====")
    print("OK  :", ok)
    print("FAIL:", fail)

if __name__ == "__main__":
    main()
