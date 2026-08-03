import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.build_pair_rule_common import (  # noqa: E402
    RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW,
    RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW_STABLE,
    public_rows,
    safe_float,
    safe_int,
    save_json,
)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path, log_path: Path):
    ensure_dir(log_path.parent)
    print(">>", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT)
        proc.wait()
        return proc.returncode


def artifact_is_fresh(path: Path, start_time: float):
    return path.exists() and path.stat().st_mtime >= start_time


def quantile(values, q):
    finite = sorted(
        safe_float(value)
        for value in values
        if math.isfinite(safe_float(value))
    )
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return float(finite[0])
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    if lo == hi:
        return float(finite[lo])
    return float(finite[lo] * (1.0 - frac) + finite[hi] * frac)


def normalize_linear(value, low, high):
    value = safe_float(value, default=float("nan"))
    low = safe_float(low, default=float("nan"))
    high = safe_float(high, default=float("nan"))
    if not math.isfinite(value) or not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return 0.0
    return max(0.0, min(1.0, float((value - low) / (high - low))))


def load_rows(path: Path):
    payload = read_json(path)
    if isinstance(payload, list):
        return payload, payload
    return payload, payload.get("rows") or []


def find_meta_path(protocol_dir: Path, pair_id: str, shift_level: str):
    direct = protocol_dir / f"{pair_id}__{shift_level}" / "split_metadata.json"
    if direct.exists():
        return direct
    nested = list(protocol_dir.glob(f"**/{pair_id}__{shift_level}/split_metadata.json"))
    if nested:
        return nested[0]
    rank_prefixed = list(protocol_dir.glob(f"**/*__{pair_id}__{shift_level}/split_metadata.json"))
    if rank_prefixed:
        return rank_prefixed[0]
    return None


def enrich_rows(rows, protocol_dir: Path):
    enriched = []
    for row in rows:
        row = dict(row)
        pair_id = row.get("pair_id")
        shift_level = row.get("shift_level") or row.get("selected_shift_level")
        row["shift_level"] = shift_level
        meta_path = None
        if pair_id and shift_level:
            meta_path = find_meta_path(protocol_dir, pair_id, shift_level)
        meta = read_json(meta_path) if meta_path and meta_path.exists() else {}
        if "target_pool_hidden_anomaly_ratio" not in row or row.get("target_pool_hidden_anomaly_ratio") in ("", None):
            row["target_pool_hidden_anomaly_ratio"] = meta.get("target_pool_hidden_anomaly_ratio")
        if "val_count" not in row or row.get("val_count") in ("", None):
            row["val_count"] = meta.get("val_count")
        if "val_anomaly_count" not in row or row.get("val_anomaly_count") in ("", None):
            row["val_anomaly_count"] = meta.get("val_anomaly_count")
        if "pad_value" not in row or row.get("pad_value") in ("", None):
            row["pad_value"] = (meta.get("candidate_pair_shift_precheck") or {}).get("pad_value")
        if "pad_feature_mean_l2" not in row or row.get("pad_feature_mean_l2") in ("", None):
            row["pad_feature_mean_l2"] = (meta.get("candidate_pair_shift_precheck") or {}).get("feature_mean_l2")
        if "build_success" not in row:
            row["build_success"] = bool(meta)
        val_count = safe_int(row.get("val_count"), default=0)
        val_anom = safe_int(row.get("val_anomaly_count"), default=0)
        if "val_anomaly_ratio" not in row or row.get("val_anomaly_ratio") in ("", None):
            row["val_anomaly_ratio"] = float(val_anom / val_count) if val_count > 0 else float("nan")
        row["_meta_path"] = str(meta_path) if meta_path else ""
        enriched.append(row)
    return enriched


def build_probe_cmd(python_exe: str, split_dir: Path, args):
    train_path = os.path.relpath(str(split_dir / "train_normal.npz"), str(PROJ_ROOT))
    val_path = os.path.relpath(str(split_dir / "val_mixed.npz"), str(PROJ_ROOT))
    dataset_arg = f"{train_path},{val_path}"
    return [
        python_exe,
        "-m",
        "src.pipeline",
        "--dataset_or_paths",
        dataset_arg,
        "--mode",
        "uad_source",
        "--family",
        "default_nasade",
        "--epochs_pretrain",
        str(args.probe_epochs_pretrain),
        "--search_candidates",
        str(args.probe_search_candidates),
        "--nas_search_iters",
        str(args.probe_nas_search_iters),
        "--nas_search_strategy",
        str(args.probe_nas_search_strategy),
        "--nas_evo_parent_pool",
        str(args.probe_nas_evo_parent_pool),
        "--nas_evo_anchor_ratio",
        str(args.probe_nas_evo_anchor_ratio),
        "--nas_evo_mutation_steps",
        str(args.probe_nas_evo_mutation_steps),
        "--nas_evo_cross_family_ratio",
        str(args.probe_nas_evo_cross_family_ratio),
        "--nas_evo_random_ratio",
        str(args.probe_nas_evo_random_ratio),
        "--batch_size",
        str(args.probe_batch_size),
        "--device",
        str(args.probe_device),
        "--seed",
        str(args.seed),
        "--oneclass_method",
        str(args.probe_oneclass_method),
        "--oneclass_epochs",
        str(args.probe_oneclass_epochs),
        "--oneclass_final_epochs",
        str(args.probe_oneclass_final_epochs),
        "--oneclass_lr",
        str(args.probe_oneclass_lr),
        "--oneclass_batch_size",
        str(args.probe_oneclass_batch_size),
        "--oneclass_max_fit",
        str(args.probe_oneclass_max_fit),
        "--knn_k",
        str(args.probe_knn_k),
        "--ocsvm_nu",
        str(args.probe_ocsvm_nu),
        "--ocsvm_kernel",
        str(args.probe_ocsvm_kernel),
        "--ocsvm_gamma",
        str(args.probe_ocsvm_gamma),
        "--ocsvm_degree",
        str(args.probe_ocsvm_degree),
        "--ocsvm_coef0",
        str(args.probe_ocsvm_coef0),
        "--svdd_hidden_dim",
        str(args.probe_svdd_hidden_dim),
        "--svdd_rep_dim",
        str(args.probe_svdd_rep_dim),
        "--svdd_nu",
        str(args.probe_svdd_nu),
        "--svdd_warmup_epochs",
        str(args.probe_svdd_warmup_epochs),
        "--svdd_final_warmup_epochs",
        str(args.probe_svdd_final_warmup_epochs),
        "--ae_hidden_dim",
        str(args.probe_ae_hidden_dim),
        "--ae_latent_dim",
        str(args.probe_ae_latent_dim),
        "--maha_hidden_dim",
        str(args.probe_maha_hidden_dim),
        "--maha_rep_dim",
        str(args.probe_maha_rep_dim),
        "--maha_shrinkage",
        str(args.probe_maha_shrinkage),
        "--gmm_hidden_dim",
        str(args.probe_gmm_hidden_dim),
        "--gmm_rep_dim",
        str(args.probe_gmm_rep_dim),
        "--gmm_components",
        str(args.probe_gmm_components),
        "--gmm_covariance_type",
        str(args.probe_gmm_covariance_type),
        "--gmm_reg_covar",
        str(args.probe_gmm_reg_covar),
        "--gmm_warmup_epochs",
        str(args.probe_gmm_warmup_epochs),
        "--proto_hidden_dim",
        str(args.probe_proto_hidden_dim),
        "--proto_rep_dim",
        str(args.probe_proto_rep_dim),
        "--proto_count",
        str(args.probe_proto_count),
        "--proto_separation_weight",
        str(args.probe_proto_separation_weight),
        "--proto_separation_margin",
        str(args.probe_proto_separation_margin),
    ]


def run_source_probe(row: dict, protocol_dir: Path, probe_root: Path, args):
    pair_id = row["pair_id"]
    shift_level = row["shift_level"]
    probe_dir = probe_root / f"{pair_id}__{shift_level}"
    ensure_dir(probe_dir)
    result_path = probe_dir / "probe_results.json"
    log_path = probe_dir / "probe.log"
    if result_path.exists() and not args.probe_force:
        return read_json(result_path)

    meta_path = find_meta_path(protocol_dir, pair_id, shift_level)
    if meta_path is None:
        payload = {
            "pair_id": pair_id,
            "shift_level": shift_level,
            "probe_success": False,
            "failure_reason": "missing_split_metadata",
        }
        save_json(result_path, payload)
        return payload

    split_dir = meta_path.parent
    train_path = split_dir / "train_normal.npz"
    val_path = split_dir / "val_mixed.npz"
    if not train_path.exists() or not val_path.exists():
        payload = {
            "pair_id": pair_id,
            "shift_level": shift_level,
            "probe_success": False,
            "failure_reason": "missing_probe_npz",
        }
        save_json(result_path, payload)
        return payload

    cmd = build_probe_cmd(args.python_exe, split_dir, args)
    start_time = time.time()
    code = run_cmd(cmd, PROJ_ROOT, log_path)
    if code != 0:
        payload = {
            "pair_id": pair_id,
            "shift_level": shift_level,
            "probe_success": False,
            "failure_reason": f"pipeline_exit_{code}",
            "log_path": str(log_path),
        }
        save_json(result_path, payload)
        return payload

    root_results = PROJ_ROOT / "outputs" / "results.json"
    if not artifact_is_fresh(root_results, start_time):
        payload = {
            "pair_id": pair_id,
            "shift_level": shift_level,
            "probe_success": False,
            "failure_reason": "missing_fresh_outputs_results",
            "log_path": str(log_path),
        }
        save_json(result_path, payload)
        return payload

    result = read_json(root_results)
    metrics = result.get("metrics_uad") or {}
    payload = {
        "pair_id": pair_id,
        "shift_level": shift_level,
        "probe_success": True,
        "best_arch": result.get("best_arch"),
        "selection_split": result.get("selection_split"),
        "report_split": result.get("report_split"),
        "source_probe_val_metrics_uad": metrics,
        "source_probe_val_auroc": safe_float(metrics.get("auroc"), default=float("nan")),
        "source_probe_val_auprc": safe_float(metrics.get("ap"), default=float("nan")),
        "source_probe_val_f1_best": safe_float(metrics.get("f1_best"), default=float("nan")),
        "log_path": str(log_path),
    }
    save_json(result_path, payload)
    return payload


def run_all_probes(rows, protocol_dir: Path, probe_root: Path, args):
    buildable_rows = [row for row in rows if row.get("build_success")]
    print(f"[INFO] Running/reading source probes for {len(buildable_rows)} buildable pairs...")
    probes = {}
    for idx, row in enumerate(buildable_rows, start=1):
        print(f"[PROBE {idx}/{len(buildable_rows)}] {row['pair_id']} [{row['shift_level']}]")
        probes[row["pair_id"]] = run_source_probe(row, protocol_dir, probe_root, args)
    return probes


def compute_thresholds(rows, args):
    successful = [row for row in rows if row.get("build_success")]
    common = {
        "min_pad_value": max(
            args.min_pad_floor,
            quantile([row.get("pad_value") for row in successful], args.min_pad_quantile),
        ),
        "max_pool_anom_ratio": min(
            args.max_pool_anom_ratio_cap,
            quantile([row.get("target_pool_hidden_anomaly_ratio") for row in successful], args.max_pool_anom_ratio_quantile),
        ),
        "min_val_count": max(
            args.min_val_floor,
            int(math.ceil(quantile([row.get("val_count") for row in successful], args.min_val_count_quantile))),
        ),
        "min_val_anomaly_count": max(
            args.min_val_anom_floor,
            int(math.ceil(quantile([row.get("val_anomaly_count") for row in successful], args.min_val_anom_quantile))),
        ),
        "max_val_anomaly_ratio": min(
            args.max_val_anom_ratio_cap,
            quantile([row.get("val_anomaly_ratio") for row in successful], args.max_val_anom_ratio_quantile),
        ),
    }
    common_feasible = [
        row
        for row in successful
        if safe_float(row.get("pad_value")) >= common["min_pad_value"]
        and safe_float(row.get("target_pool_hidden_anomaly_ratio")) <= common["max_pool_anom_ratio"]
        and safe_int(row.get("val_count")) >= common["min_val_count"]
        and safe_int(row.get("val_anomaly_count")) >= common["min_val_anomaly_count"]
        and safe_float(row.get("val_anomaly_ratio")) <= common["max_val_anomaly_ratio"]
        and math.isfinite(safe_float(row.get("source_probe_val_auroc")))
    ]
    probe_window = {
        "min_source_probe_val_auroc": max(
            args.source_probe_floor,
            quantile([row.get("source_probe_val_auroc") for row in common_feasible], args.source_probe_min_quantile),
        ),
        "max_source_probe_val_auroc": min(
            args.source_probe_cap,
            quantile([row.get("source_probe_val_auroc") for row in common_feasible], args.source_probe_max_quantile),
        ),
    }
    if safe_float(probe_window["min_source_probe_val_auroc"]) > safe_float(probe_window["max_source_probe_val_auroc"]):
        probe_window["min_source_probe_val_auroc"] = float(args.source_probe_floor)
        probe_window["max_source_probe_val_auroc"] = float(args.source_probe_cap)
    probe_stability = {
        "min_source_probe_val_auprc": float("nan"),
        "min_source_probe_val_f1_best": float("nan"),
    }
    if args.source_probe_auprc_floor is not None:
        probe_stability["min_source_probe_val_auprc"] = float(args.source_probe_auprc_floor)
    elif args.source_probe_auprc_min_quantile is not None:
        probe_stability["min_source_probe_val_auprc"] = quantile(
            [row.get("source_probe_val_auprc") for row in common_feasible],
            args.source_probe_auprc_min_quantile,
        )
    if args.source_probe_f1_best_floor is not None:
        probe_stability["min_source_probe_val_f1_best"] = float(args.source_probe_f1_best_floor)
    elif args.source_probe_f1_best_min_quantile is not None:
        probe_stability["min_source_probe_val_f1_best"] = quantile(
            [row.get("source_probe_val_f1_best") for row in common_feasible],
            args.source_probe_f1_best_min_quantile,
        )
    return {"common": common, "probe_window": probe_window, "probe_stability": probe_stability}


def passes_common(row, thresholds):
    if not row.get("build_success"):
        return False
    if not row.get("probe_success"):
        return False
    common = thresholds["common"]
    if safe_float(row.get("pad_value")) < common["min_pad_value"]:
        return False
    if safe_float(row.get("target_pool_hidden_anomaly_ratio")) > common["max_pool_anom_ratio"]:
        return False
    if safe_int(row.get("val_count")) < common["min_val_count"]:
        return False
    if safe_int(row.get("val_anomaly_count")) < common["min_val_anomaly_count"]:
        return False
    if safe_float(row.get("val_anomaly_ratio")) > common["max_val_anomaly_ratio"]:
        return False
    return True


def compute_scores(common_rows):
    if not common_rows:
        return
    pad_vals = [safe_float(row.get("pad_value")) for row in common_rows]
    pool_vals = [safe_float(row.get("target_pool_hidden_anomaly_ratio")) for row in common_rows]
    val_count_vals = [safe_float(row.get("val_count")) for row in common_rows]
    val_anom_vals = [safe_float(row.get("val_anomaly_count")) for row in common_rows]
    room_vals = [1.0 - safe_float(row.get("source_probe_val_auroc")) for row in common_rows]

    pad_lo, pad_hi = min(pad_vals), max(pad_vals)
    pool_lo, pool_hi = min(pool_vals), max(pool_vals)
    val_count_lo, val_count_hi = min(val_count_vals), max(val_count_vals)
    val_anom_lo, val_anom_hi = min(val_anom_vals), max(val_anom_vals)
    room_lo, room_hi = min(room_vals), max(room_vals)

    for row in common_rows:
        pad_n = normalize_linear(row.get("pad_value"), pad_lo, pad_hi)
        room_n = normalize_linear(1.0 - safe_float(row.get("source_probe_val_auroc")), room_lo, room_hi)
        val_count_n = normalize_linear(row.get("val_count"), val_count_lo, val_count_hi)
        val_anom_n = normalize_linear(row.get("val_anomaly_count"), val_anom_lo, val_anom_hi)
        clean_n = 1.0 - normalize_linear(row.get("target_pool_hidden_anomaly_ratio"), pool_lo, pool_hi)
        row["adaptation_window_score"] = float(
            0.40 * pad_n
            + 0.25 * room_n
            + 0.15 * val_count_n
            + 0.15 * val_anom_n
            + 0.05 * clean_n
        )


def passes_probe_stability(row, thresholds):
    probe_stability = thresholds.get("probe_stability") or {}
    min_auprc = safe_float(probe_stability.get("min_source_probe_val_auprc"), default=float("nan"))
    if math.isfinite(min_auprc) and safe_float(row.get("source_probe_val_auprc"), default=float("-inf")) < min_auprc:
        return False
    min_f1 = safe_float(probe_stability.get("min_source_probe_val_f1_best"), default=float("nan"))
    if math.isfinite(min_f1) and safe_float(row.get("source_probe_val_f1_best"), default=float("-inf")) < min_f1:
        return False
    return True


def build_rule_text(thresholds):
    common = thresholds["common"]
    probe = thresholds["probe_window"]
    probe_stability = thresholds.get("probe_stability") or {}
    extras = []
    min_auprc = safe_float(probe_stability.get("min_source_probe_val_auprc"), default=float("nan"))
    if math.isfinite(min_auprc):
        extras.append(f"source_probe_val_AUPRC >= {min_auprc:.4f}")
    min_f1 = safe_float(probe_stability.get("min_source_probe_val_f1_best"), default=float("nan"))
    if math.isfinite(min_f1):
        extras.append(f"source_probe_val_F1_best >= {min_f1:.4f}")
    extra_text = ""
    if extras:
        extra_text = " Probe stability: " + ", ".join(extras) + ". "
    return (
        "Common filters: "
        f"PAD >= {common['min_pad_value']:.4f}, "
        f"target_pool_hidden_anomaly_ratio <= {common['max_pool_anom_ratio']:.4f}, "
        f"val_count >= {common['min_val_count']}, "
        f"val_anomaly_count >= {common['min_val_anomaly_count']}, "
        f"val_anomaly_ratio <= {common['max_val_anomaly_ratio']:.4f}. "
        "Adaptation window: "
        f"{probe['min_source_probe_val_auroc']:.4f} <= source_probe_val_AUROC <= {probe['max_source_probe_val_auroc']:.4f}. "
        f"{extra_text}"
        "Ranking score = 0.40*PAD + 0.25*(1-source_probe_val_AUROC) + 0.15*val_count + "
        "0.15*val_anomaly_count + 0.05*(1-target_pool_hidden_anomaly_ratio), each normalized within the common-feasible set."
    )


def write_report(path: Path, title: str, thresholds: dict, selected_rows, all_rows):
    common_count = sum(1 for row in all_rows if row.get("passes_common"))
    probe_window_count = sum(1 for row in all_rows if row.get("in_probe_window"))
    stability_count = sum(1 for row in all_rows if row.get("passes_probe_stability"))
    lines = [
        f"# {title}",
        "",
        "## Rule",
        "",
        f"`{build_rule_text(thresholds)}`",
        "",
        f"- Buildable rows: {sum(1 for row in all_rows if row.get('build_success'))}",
        f"- Probe success rows: {sum(1 for row in all_rows if row.get('probe_success'))}",
        f"- Common-feasible rows: {common_count}",
        f"- In adaptation window: {probe_window_count}",
        f"- Passing probe-stability filter: {stability_count}",
        f"- Selected rows: {len(selected_rows)}",
        "",
        "## Selected Pairs",
        "",
        "| Order | PAD rank | Pair | Shift | PAD | source_probe_val_AUROC | source_probe_val_AUPRC | pool ratio | val count | val anom | val ratio | score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(selected_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("global_pad_rank", "")),
                    str(row.get("pair_id", "")),
                    str(row.get("shift_level", "")),
                    f"{safe_float(row.get('pad_value')):.4f}",
                    f"{safe_float(row.get('source_probe_val_auroc')):.4f}",
                    f"{safe_float(row.get('source_probe_val_auprc')):.4f}",
                    f"{safe_float(row.get('target_pool_hidden_anomaly_ratio')):.4f}",
                    str(safe_int(row.get("val_count"))),
                    str(safe_int(row.get("val_anomaly_count"))),
                    f"{safe_float(row.get('val_anomaly_ratio')):.4f}",
                    f"{safe_float(row.get('adaptation_window_score')):.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Select cross-entity hard pairs with an adaptation-window rule using cached source-only probes.")
    ap.add_argument("--selection_rows_json", required=True)
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Cross-Entity Hard Adaptation Window Selection")
    ap.add_argument(
        "--rule_name",
        default=RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW,
        choices=[
            RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW,
            RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW_STABLE,
        ],
    )
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--probe_root", default=None)
    ap.add_argument("--probe_force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_pad_quantile", type=float, default=0.50)
    ap.add_argument("--max_pool_anom_ratio_quantile", type=float, default=0.80)
    ap.add_argument("--min_val_count_quantile", type=float, default=0.25)
    ap.add_argument("--min_val_anom_quantile", type=float, default=0.15)
    ap.add_argument("--max_val_anom_ratio_quantile", type=float, default=0.90)
    ap.add_argument("--source_probe_min_quantile", type=float, default=0.20)
    ap.add_argument("--source_probe_max_quantile", type=float, default=0.80)
    ap.add_argument("--source_probe_auprc_floor", type=float, default=None)
    ap.add_argument("--source_probe_auprc_min_quantile", type=float, default=None)
    ap.add_argument("--source_probe_f1_best_floor", type=float, default=None)
    ap.add_argument("--source_probe_f1_best_min_quantile", type=float, default=None)
    ap.add_argument("--min_pad_floor", type=float, default=1.0)
    ap.add_argument("--max_pool_anom_ratio_cap", type=float, default=0.10)
    ap.add_argument("--min_val_floor", type=int, default=32)
    ap.add_argument("--min_val_anom_floor", type=int, default=7)
    ap.add_argument("--max_val_anom_ratio_cap", type=float, default=0.45)
    ap.add_argument("--source_probe_floor", type=float, default=0.55)
    ap.add_argument("--source_probe_cap", type=float, default=0.90)
    ap.add_argument("--max_selected", type=int, default=0, help="0 keeps all eligible pairs after adaptation-window filtering.")

    ap.add_argument("--probe_epochs_pretrain", type=int, default=6)
    ap.add_argument("--probe_search_candidates", type=int, default=5)
    ap.add_argument("--probe_nas_search_iters", type=int, default=2)
    ap.add_argument("--probe_nas_search_strategy", default="evolutionary_guided")
    ap.add_argument("--probe_nas_evo_parent_pool", type=int, default=3)
    ap.add_argument("--probe_nas_evo_anchor_ratio", type=float, default=0.2)
    ap.add_argument("--probe_nas_evo_mutation_steps", type=int, default=3)
    ap.add_argument("--probe_nas_evo_cross_family_ratio", type=float, default=0.5)
    ap.add_argument("--probe_nas_evo_random_ratio", type=float, default=0.2)
    ap.add_argument("--probe_batch_size", type=int, default=64)
    ap.add_argument("--probe_device", default="cpu")
    ap.add_argument("--probe_oneclass_method", default="deepsvdd")
    ap.add_argument("--probe_oneclass_epochs", type=int, default=5)
    ap.add_argument("--probe_oneclass_final_epochs", type=int, default=10)
    ap.add_argument("--probe_oneclass_lr", type=float, default=1e-3)
    ap.add_argument("--probe_oneclass_batch_size", type=int, default=1024)
    ap.add_argument("--probe_oneclass_max_fit", type=int, default=5000)
    ap.add_argument("--probe_knn_k", type=int, default=5)
    ap.add_argument("--probe_ocsvm_nu", type=float, default=0.05)
    ap.add_argument("--probe_ocsvm_kernel", default="rbf")
    ap.add_argument("--probe_ocsvm_gamma", default="scale")
    ap.add_argument("--probe_ocsvm_degree", type=int, default=3)
    ap.add_argument("--probe_ocsvm_coef0", type=float, default=0.0)
    ap.add_argument("--probe_svdd_hidden_dim", type=int, default=128)
    ap.add_argument("--probe_svdd_rep_dim", type=int, default=64)
    ap.add_argument("--probe_svdd_nu", type=float, default=0.05)
    ap.add_argument("--probe_svdd_warmup_epochs", type=int, default=2)
    ap.add_argument("--probe_svdd_final_warmup_epochs", type=int, default=5)
    ap.add_argument("--probe_ae_hidden_dim", type=int, default=128)
    ap.add_argument("--probe_ae_latent_dim", type=int, default=64)
    ap.add_argument("--probe_maha_hidden_dim", type=int, default=128)
    ap.add_argument("--probe_maha_rep_dim", type=int, default=64)
    ap.add_argument("--probe_maha_shrinkage", type=float, default=1e-2)
    ap.add_argument("--probe_gmm_hidden_dim", type=int, default=128)
    ap.add_argument("--probe_gmm_rep_dim", type=int, default=64)
    ap.add_argument("--probe_gmm_components", type=int, default=3)
    ap.add_argument("--probe_gmm_covariance_type", default="diag")
    ap.add_argument("--probe_gmm_reg_covar", type=float, default=1e-4)
    ap.add_argument("--probe_gmm_warmup_epochs", type=int, default=2)
    ap.add_argument("--probe_proto_hidden_dim", type=int, default=128)
    ap.add_argument("--probe_proto_rep_dim", type=int, default=64)
    ap.add_argument("--probe_proto_count", type=int, default=4)
    ap.add_argument("--probe_proto_separation_weight", type=float, default=0.1)
    ap.add_argument("--probe_proto_separation_margin", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()
    selection_rows_path = Path(args.selection_rows_json)
    protocol_dir = Path(args.protocol_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    probe_root = Path(args.probe_root) if args.probe_root else (out_dir / "source_probes")

    payload, rows = load_rows(selection_rows_path)
    rows = enrich_rows(rows, protocol_dir)
    probes = run_all_probes(rows, protocol_dir, probe_root, args)

    for row in rows:
        probe = probes.get(row["pair_id"]) or {}
        row["probe_success"] = bool(probe.get("probe_success"))
        row["probe_failure_reason"] = probe.get("failure_reason", "")
        row["source_probe_val_auroc"] = safe_float(probe.get("source_probe_val_auroc"), default=float("nan"))
        row["source_probe_val_auprc"] = safe_float(probe.get("source_probe_val_auprc"), default=float("nan"))
        row["source_probe_val_f1_best"] = safe_float(probe.get("source_probe_val_f1_best"), default=float("nan"))
        row["source_probe_best_arch"] = probe.get("best_arch")

    thresholds = compute_thresholds(rows, args)

    common_rows = []
    selected_rows = []
    probe_min = thresholds["probe_window"]["min_source_probe_val_auroc"]
    probe_max = thresholds["probe_window"]["max_source_probe_val_auroc"]
    for row in rows:
        row["passes_common"] = passes_common(row, thresholds)
        row["in_probe_window"] = False
        row["passes_probe_stability"] = False
        row["selected"] = False
        row["adaptation_window_score"] = None
        if row["passes_common"]:
            common_rows.append(row)
            probe_auroc = safe_float(row.get("source_probe_val_auroc"), default=float("nan"))
            row["in_probe_window"] = math.isfinite(probe_auroc) and probe_min <= probe_auroc <= probe_max
            row["passes_probe_stability"] = passes_probe_stability(row, thresholds)

    compute_scores(common_rows)
    selected_rows = [
        row for row in common_rows
        if row.get("in_probe_window") and row.get("passes_probe_stability")
    ]
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (
            -safe_float(row.get("adaptation_window_score"), default=float("-inf")),
            safe_int(row.get("global_pad_rank"), default=10**9),
            str(row.get("pair_id", "")),
        ),
    )
    if args.max_selected and args.max_selected > 0:
        selected_rows = selected_rows[: args.max_selected]
    for row in selected_rows:
        row["selected"] = True

    manifest = []
    missing = []
    rule_text = build_rule_text(thresholds)
    for order, row in enumerate(selected_rows, start=1):
        meta_path = find_meta_path(protocol_dir, row["pair_id"], row["shift_level"])
        if meta_path is None:
            missing.append(f"{row['pair_id']}::{row['shift_level']}")
            continue
        meta = read_json(meta_path)
        meta["selection_order"] = order
        meta["selected_shift_level"] = row["shift_level"]
        meta["adaptation_window"] = {
            "rule_name": args.rule_name,
            "source_probe_val_auroc": row["source_probe_val_auroc"],
            "source_probe_val_auprc": row["source_probe_val_auprc"],
            "source_probe_val_f1_best": row["source_probe_val_f1_best"],
            "source_probe_best_arch": row["source_probe_best_arch"],
            "adaptation_window_score": row["adaptation_window_score"],
            "thresholds": thresholds,
            "rule_text": rule_text,
        }
        manifest.append(meta)

    summary = {
        "source_rows_json": str(selection_rows_path),
        "protocol_dir": str(protocol_dir),
        "rule_name": args.rule_name,
        "rule_text": rule_text,
        "thresholds": thresholds,
        "probe_root": str(probe_root),
        "probe_config": {
            "epochs_pretrain": args.probe_epochs_pretrain,
            "search_candidates": args.probe_search_candidates,
            "nas_search_iters": args.probe_nas_search_iters,
            "nas_search_strategy": args.probe_nas_search_strategy,
            "device": args.probe_device,
            "oneclass_method": args.probe_oneclass_method,
            "oneclass_epochs": args.probe_oneclass_epochs,
            "oneclass_final_epochs": args.probe_oneclass_final_epochs,
        },
        "selected_count": len(selected_rows),
        "missing_meta_count": len(missing),
        "missing_meta": missing,
        "rows": public_rows(rows),
    }

    save_json(out_dir / "manifest.json", manifest)
    save_json(out_dir / "selection_summary.json", summary)
    write_report(out_dir / "REPORT.md", args.title, thresholds, selected_rows, rows)

    print(f"[DONE] Saved manifest: {out_dir / 'manifest.json'}")
    print(f"[DONE] Selected pairs: {len(selected_rows)}")
    print(f"[DONE] Rule: {rule_text}")


if __name__ == "__main__":
    main()
