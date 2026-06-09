import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace


THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.make_uad_smd import create_dataset


DEFAULT_CASES = {
    "H1": {
        "role": "high_shift_anchor",
        "source_machine": "machine-1-1",
        "target_machine": "machine-1-3",
        "family": "machine-1",
        "pad_latent_pilot": 2.0,
    },
    "L1": {
        "role": "low_shift_control",
        "source_machine": "machine-1-3",
        "target_machine": "machine-1-7",
        "family": "machine-1",
        "pad_latent_pilot": 1.5846994535519126,
    },
    "H2": {
        "role": "high_shift_extra",
        "source_machine": "machine-2-1",
        "target_machine": "machine-2-4",
        "family": "machine-2",
        "pad_latent_pilot": 2.0,
    },
    "H3": {
        "role": "high_shift_extra",
        "source_machine": "machine-1-1",
        "target_machine": "machine-1-4",
        "family": "machine-1",
        "pad_latent_pilot": 2.0,
    },
}

KEY_METRICS = [
    "auroc",
    "ap",
    "f1_best",
    "f1_pot",
    "event_f1",
    "delay_mean",
]


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path, log_path: Path):
    ensure_dir(log_path.parent)
    print(">>", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT)
        proc.wait()
        return proc.returncode


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def float_or_none(value):
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def mean_std(values: list[float]):
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {"mean": float(mean(values)), "std": float(pstdev(values)), "n": len(values)}


def case_run_name(case_id: str, case_info: dict) -> str:
    return f"{case_id}_{case_info['source_machine']}__to__{case_info['target_machine']}"


def build_split_args(
    *,
    source_dir: Path,
    target_dir: Path,
    out_dir: Path,
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
        target_machine_dir=str(target_dir),
        source_name="source.npz",
        target_name="target.npz",
        out_dir=str(out_dir),
        out_train="train_normal.npz",
        out_target_pool="target_pool_unlabeled.npz",
        out_val="val_mixed.npz",
        out_test="test_mixed.npz",
        out_meta="split_metadata.json",
        split_mode="search",
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


def ensure_processed_machine(py: str, raw_root: str, out_root: Path, machine: str, window: int, stride: int, force: bool):
    machine_dir = out_root / machine
    src = machine_dir / "source.npz"
    tgt = machine_dir / "target.npz"
    if not force and src.exists() and tgt.exists():
        print(f"[SKIP] processed machine ready: {machine}")
        return

    cmd = [
        py,
        str(PROJ_ROOT / "scripts" / "preprocess_smd.py"),
        "--raw_root",
        raw_root,
        "--out_root",
        str(out_root),
        "--machine",
        machine,
        "--window",
        str(window),
        "--stride",
        str(stride),
    ]
    log_path = PROJ_ROOT / "outputs" / "logs" / f"preprocess_{machine}.log"
    code = run_cmd(cmd, PROJ_ROOT, log_path)
    if code != 0:
        raise RuntimeError(f"preprocess failed for {machine}; see {log_path}")


def ensure_split(
    case_id: str,
    case_info: dict,
    processed_root: Path,
    experiments_root: Path,
    *,
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
    force: bool,
):
    split_dir = experiments_root / case_run_name(case_id, case_info)
    split_meta = split_dir / "split_metadata.json"
    if not force and split_meta.exists():
        print(f"[SKIP] split ready: {split_dir}")
        return split_dir, read_json(split_meta)

    args = build_split_args(
        source_dir=processed_root / case_info["source_machine"],
        target_dir=processed_root / case_info["target_machine"],
        out_dir=split_dir,
        shift_level=shift_level,
        target_pool_frac=target_pool_frac,
        val_frac=val_frac,
        guard=guard,
        search_step=search_step,
        max_pool_anom_ratio=max_pool_anom_ratio,
        min_target_pool=min_target_pool,
        min_val=min_val,
        min_test=min_test,
        min_anom_val=min_anom_val,
        min_anom_test=min_anom_test,
        seed=seed,
    )
    meta = create_dataset(args)
    return split_dir, meta


def dataset_arg_for_mode(split_dir: Path, mode: str):
    train_path = split_dir / "train_normal.npz"
    target_pool_path = split_dir / "target_pool_unlabeled.npz"
    val_path = split_dir / "val_mixed.npz"
    test_path = split_dir / "test_mixed.npz"

    if mode == "uad_source":
        parts = [train_path, val_path, test_path]
    else:
        parts = [train_path, target_pool_path, val_path, test_path]
    return ",".join(os.path.relpath(str(path), str(PROJ_ROOT)) for path in parts)


def artifact_is_fresh(path: Path, start_time: float):
    return path.exists() and path.stat().st_mtime >= start_time


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_mode_for_case_seed(
    py: str,
    split_dir: Path,
    run_dir: Path,
    *,
    mode: str,
    seed: int,
    epochs_pretrain: int,
    search_candidates: int,
    batch_size: int,
    device: str,
    combined_upper_gap: float,
    oneclass_cli_args,
    force: bool,
):
    ensure_dir(run_dir)
    result_path = run_dir / f"{mode}_results.json"
    baselines_summary_path = run_dir / f"{mode}_baselines_summary.json"
    baselines_dir = run_dir / f"{mode}_baselines"
    log_path = run_dir / f"{mode}.log"

    if not force and result_path.exists():
        if mode != "adaptnas_combined" or baselines_summary_path.exists():
            print(f"[SKIP] run ready: {run_dir.name} {mode}")
            return

    cmd = [
        py,
        "-m",
        "src.pipeline",
        "--dataset_or_paths",
        dataset_arg_for_mode(split_dir, mode),
        "--mode",
        mode,
        "--family",
        "default_nasade",
        "--epochs_pretrain",
        str(epochs_pretrain),
        "--search_candidates",
        str(search_candidates),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--combined_upper_gap",
        str(combined_upper_gap),
        "--seed",
        str(seed),
    ]
    cmd.extend(oneclass_cli_args)

    start_time = time.time()
    code = run_cmd(cmd, PROJ_ROOT, log_path)
    if code != 0:
        raise RuntimeError(f"pipeline failed for {mode} at seed {seed}; see {log_path}")

    root_results = PROJ_ROOT / "outputs" / "results.json"
    if not artifact_is_fresh(root_results, start_time):
        raise RuntimeError(f"missing fresh outputs/results.json after {mode}; see {log_path}")
    shutil.copy2(root_results, result_path)

    if mode == "adaptnas_combined":
        root_summary = PROJ_ROOT / "outputs" / "baselines_summary.json"
        root_baselines = PROJ_ROOT / "outputs" / "baselines"
        if not artifact_is_fresh(root_summary, start_time):
            raise RuntimeError(f"missing fresh outputs/baselines_summary.json after {mode}; see {log_path}")
        shutil.copy2(root_summary, baselines_summary_path)
        if root_baselines.exists():
            copy_tree(root_baselines, baselines_dir)


def extract_combined_baseline_info(summary: dict):
    all_entries = summary.get("all") or []
    by_arch = {}
    for entry in all_entries:
        arch_name = entry.get("arch_name")
        if arch_name:
            by_arch[arch_name] = entry.get("metrics_uad") or {}

    best_fixed_name = None
    best_fixed_metrics = None
    best_fixed_auroc = float("-inf")
    for arch_name, metrics in by_arch.items():
        if not arch_name.startswith("Base_"):
            continue
        auroc = float_or_none(metrics.get("auroc"))
        if auroc is not None and auroc > best_fixed_auroc:
            best_fixed_auroc = auroc
            best_fixed_name = arch_name
            best_fixed_metrics = metrics

    best_by_auroc = summary.get("best_by_auroc") or {}
    return {
        "winner_arch": best_by_auroc.get("arch_name"),
        "winner_metrics": best_by_auroc.get("metrics_uad") or {},
        "nas_metrics": by_arch.get("NAS_BestArch") or {},
        "best_fixed_arch": best_fixed_name,
        "best_fixed_metrics": best_fixed_metrics or {},
        "all_by_arch": by_arch,
    }


def build_seed_row(case_id: str, case_info: dict, seed: int, split_meta: dict, source_res: dict, combined_res: dict, combined_summary: dict):
    source_metrics = source_res.get("metrics_uad") or {}
    combined_metrics = combined_res.get("metrics_uad") or {}
    baseline_info = extract_combined_baseline_info(combined_summary)
    best_fixed_metrics = baseline_info["best_fixed_metrics"]
    nas_metrics = baseline_info["nas_metrics"]

    row = {
        "case_id": case_id,
        "role": case_info["role"],
        "family": case_info["family"],
        "source_machine": case_info["source_machine"],
        "target_machine": case_info["target_machine"],
        "pair": f"{case_info['source_machine']} -> {case_info['target_machine']}",
        "seed": seed,
        "pad_latent_pilot": float(case_info["pad_latent_pilot"]),
        "split_out_dir": split_meta.get("out_dir") if split_meta else None,
        "combined_winner_arch": baseline_info["winner_arch"],
        "best_fixed_arch": baseline_info["best_fixed_arch"],
    }

    for metric in KEY_METRICS:
        src_val = float_or_none(source_metrics.get(metric))
        cmb_val = float_or_none(combined_metrics.get(metric))
        fixed_val = float_or_none(best_fixed_metrics.get(metric))
        nas_val = float_or_none(nas_metrics.get(metric))

        row[f"uad_source_{metric}"] = src_val
        row[f"adaptnas_combined_{metric}"] = cmb_val
        row[f"best_fixed_{metric}"] = fixed_val
        row[f"nas_bestarch_{metric}"] = nas_val

        if src_val is not None and cmb_val is not None:
            row[f"delta_combined_minus_source_{metric}"] = cmb_val - src_val
        else:
            row[f"delta_combined_minus_source_{metric}"] = None

        if fixed_val is not None and nas_val is not None:
            row[f"delta_nas_minus_best_fixed_{metric}"] = nas_val - fixed_val
        else:
            row[f"delta_nas_minus_best_fixed_{metric}"] = None

    return row


def aggregate_rows(rows: list[dict]):
    case_summary = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        summary = {
            "case_id": case_id,
            "role": case_rows[0]["role"],
            "family": case_rows[0]["family"],
            "pair": case_rows[0]["pair"],
            "pad_latent_pilot": case_rows[0]["pad_latent_pilot"],
            "num_seeds": len(case_rows),
            "seed_values": [row["seed"] for row in case_rows],
            "combined_winner_archs": sorted({str(row["combined_winner_arch"]) for row in case_rows}),
            "best_fixed_archs": sorted({str(row["best_fixed_arch"]) for row in case_rows}),
            "metrics": {},
        }
        for metric in KEY_METRICS:
            for prefix in [
                "uad_source",
                "adaptnas_combined",
                "best_fixed",
                "nas_bestarch",
                "delta_combined_minus_source",
                "delta_nas_minus_best_fixed",
            ]:
                key = f"{prefix}_{metric}"
                vals = [row[key] for row in case_rows if row.get(key) is not None]
                summary["metrics"][key] = mean_std(vals)
        case_summary[case_id] = summary

    group_summary = {}
    groups = {
        "high_shift": [row for row in rows if row["role"].startswith("high_shift")],
        "low_shift": [row for row in rows if row["role"] == "low_shift_control"],
    }
    for group_name, group_rows in groups.items():
        group_summary[group_name] = {
            "num_rows": len(group_rows),
            "metrics": {},
        }
        for metric in KEY_METRICS:
            delta_key = f"delta_combined_minus_source_{metric}"
            vals = [row[delta_key] for row in group_rows if row.get(delta_key) is not None]
            group_summary[group_name]["metrics"][delta_key] = mean_std(vals)

    auroc_high = group_summary["high_shift"]["metrics"]["delta_combined_minus_source_auroc"]["mean"]
    auroc_low = group_summary["low_shift"]["metrics"]["delta_combined_minus_source_auroc"]["mean"]
    benefit_gap = None
    if auroc_high is not None and auroc_low is not None:
        benefit_gap = auroc_high - auroc_low

    return case_summary, group_summary, benefit_gap


def write_rows_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    ensure_dir(path.parent)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_metric(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def build_summary_markdown(config: dict, rows: list[dict], case_summary: dict, group_summary: dict, benefit_gap):
    lines = []
    lines.append("# Domain Shift Case Matrix Summary")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append(f"- Cases: `{', '.join(config['cases'])}`")
    lines.append(f"- Seeds: `{', '.join(str(seed) for seed in config['seeds'])}`")
    lines.append(f"- Window/stride: `{config['window']}/{config['stride']}`")
    lines.append(f"- Split: `shift_level={config['shift_level']}`, `target_pool_frac={config['target_pool_frac']}`, `val_frac={config['val_frac']}`, `guard={config['guard']}`")
    lines.append(f"- Train/search: `epochs_pretrain={config['epochs_pretrain']}`, `search_candidates={config['search_candidates']}`, `batch_size={config['batch_size']}`, `device={config['device']}`, `seed={config['seeds'][0]}`")
    lines.append("")

    lines.append("## Pair-Seed Comparison")
    lines.append("")
    lines.append("| Case | Role | Pair | Seed | PAD | Source AUROC | Combined AUROC | Delta AUROC | Source F1_best | Combined F1_best | Delta F1_best | Combined winner | Best fixed baseline | NAS AUROC | Best fixed AUROC |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|")
    for row in sorted(rows, key=lambda item: (item["case_id"], item["seed"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["role"],
                    row["pair"],
                    str(row["seed"]),
                    fmt_metric(row["pad_latent_pilot"]),
                    fmt_metric(row["uad_source_auroc"]),
                    fmt_metric(row["adaptnas_combined_auroc"]),
                    fmt_metric(row["delta_combined_minus_source_auroc"]),
                    fmt_metric(row["uad_source_f1_best"]),
                    fmt_metric(row["adaptnas_combined_f1_best"]),
                    fmt_metric(row["delta_combined_minus_source_f1_best"]),
                    str(row["combined_winner_arch"]),
                    str(row["best_fixed_arch"]),
                    fmt_metric(row["nas_bestarch_auroc"]),
                    fmt_metric(row["best_fixed_auroc"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Case Aggregates")
    lines.append("")
    lines.append("| Case | Pair | Seeds | Source AUROC mean+-std | Combined AUROC mean+-std | Delta AUROC mean+-std | Source Event_F1 mean+-std | Combined Event_F1 mean+-std | Delta Event_F1 mean+-std | Winner archs |")
    lines.append("|---|---|---:|---|---|---|---|---|---|---|")
    for case_id in sorted(case_summary):
        item = case_summary[case_id]
        src_auroc = item["metrics"]["uad_source_auroc"]
        cmb_auroc = item["metrics"]["adaptnas_combined_auroc"]
        delta_auroc = item["metrics"]["delta_combined_minus_source_auroc"]
        src_event = item["metrics"]["uad_source_event_f1"]
        cmb_event = item["metrics"]["adaptnas_combined_event_f1"]
        delta_event = item["metrics"]["delta_combined_minus_source_event_f1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    case_id,
                    item["pair"],
                    str(item["num_seeds"]),
                    f"{fmt_metric(src_auroc['mean'])} +- {fmt_metric(src_auroc['std'])}",
                    f"{fmt_metric(cmb_auroc['mean'])} +- {fmt_metric(cmb_auroc['std'])}",
                    f"{fmt_metric(delta_auroc['mean'])} +- {fmt_metric(delta_auroc['std'])}",
                    f"{fmt_metric(src_event['mean'])} +- {fmt_metric(src_event['std'])}",
                    f"{fmt_metric(cmb_event['mean'])} +- {fmt_metric(cmb_event['std'])}",
                    f"{fmt_metric(delta_event['mean'])} +- {fmt_metric(delta_event['std'])}",
                    ", ".join(item["combined_winner_archs"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Group Summary")
    lines.append("")
    lines.append("| Group | Delta AUROC mean+-std | Delta F1_best mean+-std | Delta Event_F1 mean+-std | Delta Delay_mean mean+-std |")
    lines.append("|---|---|---|---|---|")
    for group_name in ["high_shift", "low_shift"]:
        group_item = group_summary[group_name]["metrics"]
        auroc_delta = group_item["delta_combined_minus_source_auroc"]
        f1_delta = group_item["delta_combined_minus_source_f1_best"]
        event_delta = group_item["delta_combined_minus_source_event_f1"]
        delay_delta = group_item["delta_combined_minus_source_delay_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    f"{fmt_metric(auroc_delta['mean'])} +- {fmt_metric(auroc_delta['std'])}",
                    f"{fmt_metric(f1_delta['mean'])} +- {fmt_metric(f1_delta['std'])}",
                    f"{fmt_metric(event_delta['mean'])} +- {fmt_metric(event_delta['std'])}",
                    f"{fmt_metric(delay_delta['mean'])} +- {fmt_metric(delay_delta['std'])}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(f"- Benefit_gap (high-shift delta AUROC minus low-shift delta AUROC): `{fmt_metric(benefit_gap)}`")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="L1,H2,H3", help="Comma-separated case ids. Defaults to the 3 new cases.")
    ap.add_argument("--include_anchor", action="store_true", help="Include existing H1 anchor results in aggregation.")
    ap.add_argument(
        "--anchor_results_dir",
        default=str(PROJ_ROOT / "outputs" / "benchmarks" / "top_pair_compare_m1_1_to_m1_3_paper_grade"),
        help="Existing H1 results directory used when --include_anchor is set.",
    )
    ap.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42 or 42,43,44")
    ap.add_argument("--raw_root", default="data/ServerMachineDataset")
    ap.add_argument("--processed_root", default=str(PROJ_ROOT / "data" / "smd"))
    ap.add_argument("--experiments_root", default=str(PROJ_ROOT / "data" / "smd_experiments" / "domain_shift_matrix"))
    ap.add_argument("--output_root", default=str(PROJ_ROOT / "outputs" / "benchmarks" / "domain_shift_matrix"))
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--shift_level", default="hard")
    ap.add_argument("--target_pool_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--guard", type=int, default=4)
    ap.add_argument("--search_step", type=int, default=4)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=0.1)
    ap.add_argument("--min_target_pool", type=int, default=32)
    ap.add_argument("--min_val", type=int, default=32)
    ap.add_argument("--min_test", type=int, default=64)
    ap.add_argument("--min_anom_val", type=int, default=3)
    ap.add_argument("--min_anom_test", type=int, default=5)
    ap.add_argument("--epochs_pretrain", type=int, default=10)
    ap.add_argument("--search_candidates", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--oneclass_method", default="deepsvdd", choices=["deepsvdd", "autoencoder", "knn_distance", "oneclass_svm", "svdd", "prototype_oneclass", "mahalanobis_head", "gmm_head"])
    ap.add_argument("--oneclass_epochs", type=int, default=10)
    ap.add_argument("--oneclass_final_epochs", type=int, default=20)
    ap.add_argument("--oneclass_lr", type=float, default=1e-3)
    ap.add_argument("--oneclass_batch_size", type=int, default=1024)
    ap.add_argument("--oneclass_max_fit", type=int, default=5000)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--ocsvm_nu", type=float, default=0.05)
    ap.add_argument("--ocsvm_kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    ap.add_argument("--ocsvm_gamma", default="scale")
    ap.add_argument("--ocsvm_degree", type=int, default=3)
    ap.add_argument("--ocsvm_coef0", type=float, default=0.0)
    ap.add_argument("--svdd_hidden_dim", type=int, default=128)
    ap.add_argument("--svdd_rep_dim", type=int, default=64)
    ap.add_argument("--svdd_nu", type=float, default=0.05)
    ap.add_argument("--svdd_warmup_epochs", type=int, default=2)
    ap.add_argument("--svdd_final_warmup_epochs", type=int, default=5)
    ap.add_argument("--ae_hidden_dim", type=int, default=128)
    ap.add_argument("--ae_latent_dim", type=int, default=64)
    ap.add_argument("--maha_hidden_dim", type=int, default=128)
    ap.add_argument("--maha_rep_dim", type=int, default=64)
    ap.add_argument("--maha_shrinkage", type=float, default=1e-2)
    ap.add_argument("--gmm_hidden_dim", type=int, default=128)
    ap.add_argument("--gmm_rep_dim", type=int, default=64)
    ap.add_argument("--gmm_components", type=int, default=3)
    ap.add_argument("--gmm_covariance_type", default="diag", choices=["diag", "full"])
    ap.add_argument("--gmm_reg_covar", type=float, default=1e-4)
    ap.add_argument("--gmm_warmup_epochs", type=int, default=2)
    ap.add_argument("--proto_hidden_dim", type=int, default=128)
    ap.add_argument("--proto_rep_dim", type=int, default=64)
    ap.add_argument("--proto_count", type=int, default=4)
    ap.add_argument("--proto_separation_weight", type=float, default=0.1)
    ap.add_argument("--proto_separation_margin", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--combined_upper_gap", type=float, default=1.0)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--force_preprocess", action="store_true")
    ap.add_argument("--force_split", action="store_true")
    ap.add_argument("--force_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    oneclass_cli_args = [
        "--oneclass_method", args.oneclass_method,
        "--oneclass_epochs", str(args.oneclass_epochs),
        "--oneclass_final_epochs", str(args.oneclass_final_epochs),
        "--oneclass_lr", str(args.oneclass_lr),
        "--oneclass_batch_size", str(args.oneclass_batch_size),
        "--oneclass_max_fit", str(args.oneclass_max_fit),
        "--knn_k", str(args.knn_k),
        "--ocsvm_nu", str(args.ocsvm_nu),
        "--ocsvm_kernel", args.ocsvm_kernel,
        "--ocsvm_gamma", str(args.ocsvm_gamma),
        "--ocsvm_degree", str(args.ocsvm_degree),
        "--ocsvm_coef0", str(args.ocsvm_coef0),
        "--svdd_hidden_dim", str(args.svdd_hidden_dim),
        "--svdd_rep_dim", str(args.svdd_rep_dim),
        "--svdd_nu", str(args.svdd_nu),
        "--svdd_warmup_epochs", str(args.svdd_warmup_epochs),
        "--svdd_final_warmup_epochs", str(args.svdd_final_warmup_epochs),
        "--ae_hidden_dim", str(args.ae_hidden_dim),
        "--ae_latent_dim", str(args.ae_latent_dim),
        "--maha_hidden_dim", str(args.maha_hidden_dim),
        "--maha_rep_dim", str(args.maha_rep_dim),
        "--maha_shrinkage", str(args.maha_shrinkage),
        "--gmm_hidden_dim", str(args.gmm_hidden_dim),
        "--gmm_rep_dim", str(args.gmm_rep_dim),
        "--gmm_components", str(args.gmm_components),
        "--gmm_covariance_type", args.gmm_covariance_type,
        "--gmm_reg_covar", str(args.gmm_reg_covar),
        "--gmm_warmup_epochs", str(args.gmm_warmup_epochs),
        "--proto_hidden_dim", str(args.proto_hidden_dim),
        "--proto_rep_dim", str(args.proto_rep_dim),
        "--proto_count", str(args.proto_count),
        "--proto_separation_weight", str(args.proto_separation_weight),
        "--proto_separation_margin", str(args.proto_separation_margin),
    ]
    processed_root = Path(args.processed_root)
    experiments_root = Path(args.experiments_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    selected_case_ids = parse_csv_arg(args.cases)
    if not selected_case_ids:
        raise ValueError("No case ids provided.")
    for case_id in selected_case_ids:
        if case_id not in DEFAULT_CASES:
            raise ValueError(f"Unknown case id: {case_id}")

    selected_case_map = {case_id: DEFAULT_CASES[case_id] for case_id in selected_case_ids}
    seeds = [int(seed) for seed in parse_csv_arg(args.seeds)]
    if not seeds:
        raise ValueError("No seeds provided.")

    machines = sorted(
        {
            machine
            for case_info in selected_case_map.values()
            for machine in (case_info["source_machine"], case_info["target_machine"])
        }
    )
    for machine in machines:
        ensure_processed_machine(
            py,
            args.raw_root,
            processed_root,
            machine,
            args.window,
            args.stride,
            args.force_preprocess,
        )

    split_metadata_by_case = {}
    for case_id, case_info in selected_case_map.items():
        split_dir, split_meta = ensure_split(
            case_id,
            case_info,
            processed_root,
            experiments_root,
            shift_level=args.shift_level,
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
            seed=args.split_seed,
            force=args.force_split,
        )
        split_metadata_by_case[case_id] = {"dir": split_dir, "meta": split_meta}

    rows = []

    if args.include_anchor:
        anchor_dir = Path(args.anchor_results_dir)
        anchor_source = anchor_dir / "uad_source_results.json"
        anchor_combined = anchor_dir / "adaptnas_combined_results.json"
        anchor_summary = anchor_dir / "adaptnas_combined_baselines_summary.json"
        if anchor_source.exists() and anchor_combined.exists() and anchor_summary.exists():
            split_meta = read_json(
                PROJ_ROOT
                / "data"
                / "smd_experiments"
                / "cross_machine_hard"
                / "machine-1-1__to__machine-1-3"
                / "split_metadata.json"
            )
            rows.append(
                build_seed_row(
                    "H1",
                    DEFAULT_CASES["H1"],
                    42,
                    split_meta,
                    read_json(anchor_source),
                    read_json(anchor_combined),
                    read_json(anchor_summary),
                )
            )
        else:
            print(f"[WARN] anchor results missing under {anchor_dir}; skipping H1 aggregation.")

    for case_id, case_info in selected_case_map.items():
        split_info = split_metadata_by_case[case_id]
        split_dir = split_info["dir"]
        split_meta = split_info["meta"]
        case_output_dir = output_root / case_run_name(case_id, case_info)
        for seed in seeds:
            run_dir = case_output_dir / f"seed_{seed}"
            run_mode_for_case_seed(
                py,
                split_dir,
                run_dir,
                mode="uad_source",
                seed=seed,
                epochs_pretrain=args.epochs_pretrain,
                search_candidates=args.search_candidates,
                batch_size=args.batch_size,
                device=args.device,
                combined_upper_gap=args.combined_upper_gap,
                oneclass_cli_args=oneclass_cli_args,
                force=args.force_run,
            )
            run_mode_for_case_seed(
                py,
                split_dir,
                run_dir,
                mode="adaptnas_combined",
                seed=seed,
                epochs_pretrain=args.epochs_pretrain,
                search_candidates=args.search_candidates,
                batch_size=args.batch_size,
                device=args.device,
                combined_upper_gap=args.combined_upper_gap,
                oneclass_cli_args=oneclass_cli_args,
                force=args.force_run,
            )

            rows.append(
                build_seed_row(
                    case_id,
                    case_info,
                    seed,
                    split_meta,
                    read_json(run_dir / "uad_source_results.json"),
                    read_json(run_dir / "adaptnas_combined_results.json"),
                    read_json(run_dir / "adaptnas_combined_baselines_summary.json"),
                )
            )

    case_summary, group_summary, benefit_gap = aggregate_rows(rows)
    summary_payload = {
        "config": {
            "cases": (["H1"] if args.include_anchor else []) + selected_case_ids,
            "seeds": seeds,
            "window": args.window,
            "stride": args.stride,
            "shift_level": args.shift_level,
            "target_pool_frac": args.target_pool_frac,
            "val_frac": args.val_frac,
            "guard": args.guard,
            "epochs_pretrain": args.epochs_pretrain,
            "search_candidates": args.search_candidates,
            "batch_size": args.batch_size,
            "oneclass_method": args.oneclass_method,
            "oneclass_epochs": args.oneclass_epochs,
            "oneclass_final_epochs": args.oneclass_final_epochs,
            "oneclass_lr": args.oneclass_lr,
            "oneclass_batch_size": args.oneclass_batch_size,
            "oneclass_max_fit": args.oneclass_max_fit,
            "knn_k": args.knn_k,
            "ocsvm_nu": args.ocsvm_nu,
            "ocsvm_kernel": args.ocsvm_kernel,
            "ocsvm_gamma": args.ocsvm_gamma,
            "ocsvm_degree": args.ocsvm_degree,
            "ocsvm_coef0": args.ocsvm_coef0,
            "svdd_hidden_dim": args.svdd_hidden_dim,
            "svdd_rep_dim": args.svdd_rep_dim,
            "svdd_nu": args.svdd_nu,
            "svdd_warmup_epochs": args.svdd_warmup_epochs,
            "svdd_final_warmup_epochs": args.svdd_final_warmup_epochs,
            "ae_hidden_dim": args.ae_hidden_dim,
            "ae_latent_dim": args.ae_latent_dim,
            "maha_hidden_dim": args.maha_hidden_dim,
            "maha_rep_dim": args.maha_rep_dim,
            "maha_shrinkage": args.maha_shrinkage,
            "gmm_hidden_dim": args.gmm_hidden_dim,
            "gmm_rep_dim": args.gmm_rep_dim,
            "gmm_components": args.gmm_components,
            "gmm_covariance_type": args.gmm_covariance_type,
            "gmm_reg_covar": args.gmm_reg_covar,
            "gmm_warmup_epochs": args.gmm_warmup_epochs,
            "proto_hidden_dim": args.proto_hidden_dim,
            "proto_rep_dim": args.proto_rep_dim,
            "proto_count": args.proto_count,
            "proto_separation_weight": args.proto_separation_weight,
            "proto_separation_margin": args.proto_separation_margin,
            "device": args.device,
            "split_seed": args.split_seed,
        },
        "rows": rows,
        "case_summary": case_summary,
        "group_summary": group_summary,
        "benefit_gap": benefit_gap,
    }

    write_json(output_root / "summary.json", summary_payload)
    write_rows_csv(output_root / "summary_rows.csv", rows)
    summary_md = build_summary_markdown(summary_payload["config"], rows, case_summary, group_summary, benefit_gap)
    (output_root / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"[DONE] Summary JSON: {output_root / 'summary.json'}")
    print(f"[DONE] Summary CSV : {output_root / 'summary_rows.csv'}")
    print(f"[DONE] Summary MD  : {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
