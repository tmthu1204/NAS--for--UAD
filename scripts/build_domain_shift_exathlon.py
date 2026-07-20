import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.build_pair_rule_common import (
    RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3,
    RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW,
    RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD,
    RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW,
    RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW,
    finite_quantile,
    load_pad_pair_rankings,
    public_rows,
    save_json,
    write_table_csv,
    write_table_markdown,
)
from scripts.make_uad_smd import (
    DEFAULT_GUARD,
    DEFAULT_MAX_POOL_ANOM_RATIO,
    DEFAULT_MIN_ANOM_TEST,
    DEFAULT_MIN_ANOM_VAL,
    DEFAULT_MIN_TARGET_POOL,
    DEFAULT_MIN_TEST,
    DEFAULT_MIN_VAL,
    DEFAULT_SEARCH_STEP,
    DEFAULT_TARGET_POOL_FRAC,
    DEFAULT_VAL_FRAC,
    binarize_y,
    compute_domain_shift_metrics,
    create_dataset,
    load_npz,
)


def list_trace_dirs(data_root: Path):
    return sorted([p for p in data_root.iterdir() if p.is_dir() and (p / "source.npz").exists() and (p / "target.npz").exists()])


def read_trace_meta(trace_dir: Path) -> dict:
    meta_path = trace_dir / "metadata.json"
    if not meta_path.exists():
        return {"app_name": trace_family(trace_dir.name), "type_id": None}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def trace_family(name: str) -> str:
    return name.split("-")[0] if "-" in name else name


def source_norm_windows(trace_dir: Path):
    Xs, ys = load_npz(str(trace_dir / "source.npz"))
    ys = binarize_y(ys)
    return Xs[ys == 0]


def target_windows(trace_dir: Path):
    Xt, yt = load_npz(str(trace_dir / "target.npz"))
    yt = binarize_y(yt)
    return Xt, yt


def is_allowed_source(trace_dir: Path, source_type_ids: set[int] | None):
    if source_type_ids is None:
        return True
    meta = read_trace_meta(trace_dir)
    type_id = meta.get("type_id")
    return type_id in source_type_ids


def candidate_cross_targets(source_dir: Path, all_dirs, same_app_only: bool):
    src_family = trace_family(source_dir.name)
    out = []
    for target_dir in all_dirs:
        if target_dir == source_dir:
            continue
        if same_app_only and trace_family(target_dir.name) != src_family:
            continue
        out.append(target_dir)
    return out


def rank_cross_targets(source_dir: Path, target_dirs, min_target_anom: int):
    Xs_norm = source_norm_windows(source_dir)
    ranked = []
    for target_dir in target_dirs:
        Xt, yt = target_windows(target_dir)
        target_anom = int((yt == 1).sum())
        if target_anom < min_target_anom:
            continue
        shift = compute_domain_shift_metrics(Xs_norm, Xt, seed=42)
        auc = shift["domain_auc"]
        if not np.isfinite(auc):
            auc = 0.5
        ranked.append((float(auc), target_anom, target_dir, shift))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def build_args(
    *,
    source_dir: Path,
    target_dir: Path | None,
    out_dir: Path,
    split_mode: str,
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
        target_machine_dir=(str(target_dir) if target_dir is not None else None),
        source_name="source.npz",
        target_name="target.npz",
        out_dir=str(out_dir),
        out_train="train_normal.npz",
        out_target_pool="target_pool_unlabeled.npz",
        out_val="val_mixed.npz",
        out_test="test_mixed.npz",
        out_meta="split_metadata.json",
        split_mode=split_mode,
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


def parse_source_type_ids(text: str):
    text = (text or "").strip()
    if not text or text.lower() == "all":
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def source_pool_pad(meta: dict) -> dict:
    shift = ((meta.get("domain_shift") or {}).get("source_vs_target_pool") or {})
    return {
        "pad_value": float(shift.get("feature_mean_l2", float("nan"))),
        "domain_acc": float(shift.get("domain_acc", float("nan"))),
        "domain_auc": float(shift.get("domain_auc", float("nan"))),
        "feature_mean_l2": float(shift.get("feature_mean_l2", float("nan"))),
    }


def run_cmd(cmd: list[str], cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(">>", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT)
        proc.wait()
        return proc.returncode


def artifact_is_fresh(path: Path, start_time: float):
    return path.exists() and path.stat().st_mtime >= start_time


def build_source_probe_cmd(split_dir: Path, args):
    train_path = os.path.relpath(str(split_dir / "train_normal.npz"), str(PROJ_ROOT))
    val_path = os.path.relpath(str(split_dir / "val_mixed.npz"), str(PROJ_ROOT))
    dataset_arg = f"{train_path},{val_path}"
    return [
        sys.executable,
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
        "--oneclass_batch_size",
        str(args.probe_oneclass_batch_size),
        "--oneclass_max_fit",
        str(args.probe_oneclass_max_fit),
    ]


def run_source_probe_for_row(protocol_dir: Path, row: dict, args):
    pair_id = row["pair_id"]
    shift_level = row["shift_level"]
    probe_root = (
        Path(args.same_app_adaptation_probe_root)
        if args.same_app_adaptation_probe_root
        else (protocol_dir / "source_probes")
    )
    probe_dir = probe_root / f"{pair_id}__{shift_level}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    result_path = probe_dir / "probe_results.json"
    log_path = probe_dir / "probe.log"
    if result_path.exists() and not args.same_app_adaptation_probe_force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    split_dir = protocol_dir / pair_id
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

    cmd = build_source_probe_cmd(split_dir, args)
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

    result = json.loads(root_results.read_text(encoding="utf-8"))
    metrics = result.get("metrics_uad") or {}
    payload = {
        "pair_id": pair_id,
        "shift_level": shift_level,
        "probe_success": True,
        "best_arch": result.get("best_arch"),
        "selection_split": result.get("selection_split"),
        "report_split": result.get("report_split"),
        "source_probe_val_metrics_uad": metrics,
        "source_probe_val_auroc": float(metrics.get("auroc", float("nan"))),
        "source_probe_val_auprc": float(metrics.get("ap", float("nan"))),
        "source_probe_val_f1_best": float(metrics.get("f1_best", float("nan"))),
        "log_path": str(log_path),
    }
    save_json(result_path, payload)
    return payload


def build_same_app_disturbed_moderate_methods_text(*, source_type_ids, target_type_ids, max_pad_quantile):
    src_text = "all" if source_type_ids is None else ",".join(str(int(v)) for v in sorted(source_type_ids))
    tgt_text = "all" if target_type_ids is None else ",".join(str(int(v)) for v in sorted(target_type_ids))
    quantile_pct = int(round(max(0.0, min(1.0, float(max_pad_quantile))) * 100))
    return (
        "We built same-application cross-run hard pairs only. "
        f"Source runs were restricted to type_id in {{{src_text}}}, while target runs were restricted to disturbed type_id in {{{tgt_text}}}. "
        f"Within the buildable disturbed subset, we excluded the extreme PAD upper tail and retained only pairs whose source-vs-target-pool feature-mean L2 fell at or below the dataset-specific Q{quantile_pct} cutoff."
    )


def build_same_app_adaptation_window_methods_text(
    *,
    source_type_ids,
    target_type_ids,
    max_pad_value,
    min_val_anomaly_ratio,
    min_train_normal_count,
    max_source_probe_val_auroc,
):
    src_text = "all" if source_type_ids is None else ",".join(str(int(v)) for v in sorted(source_type_ids))
    tgt_text = "all" if target_type_ids is None else ",".join(str(int(v)) for v in sorted(target_type_ids))
    val_pct = round(float(min_val_anomaly_ratio) * 100.0, 1)
    return (
        "We built same-application cross-run hard pairs only. "
        f"Source runs were restricted to type_id in {{{src_text}}}, while target runs were restricted to disturbed type_id in {{{tgt_text}}}. "
        "To retain only hard but still learnable shifts, we kept buildable pairs whose PAD did not exceed "
        f"{float(max_pad_value):.4f}, whose validation anomaly ratio was at least {val_pct}%, whose source train-normal support was at least "
        f"{int(min_train_normal_count)} windows, and whose lightweight source-only probe on val_mixed remained below AUROC {float(max_source_probe_val_auroc):.4f}, "
        "thereby excluding already saturated source-only cases."
    )


def build_same_app_narrow_adaptation_window_methods_text(
    *,
    source_type_ids,
    target_type_ids,
    min_pad_rank_quantile,
    max_pad_rank_quantile,
    min_val_anomaly_ratio,
    max_val_anomaly_ratio,
    min_val_anomaly_count,
    min_train_normal_count,
):
    src_text = "all" if source_type_ids is None else ",".join(str(int(v)) for v in sorted(source_type_ids))
    tgt_text = "all" if target_type_ids is None else ",".join(str(int(v)) for v in sorted(target_type_ids))
    rank_lo_pct = int(round(max(0.0, min(1.0, float(min_pad_rank_quantile))) * 100.0))
    rank_hi_pct = int(round(max(0.0, min(1.0, float(max_pad_rank_quantile))) * 100.0))
    val_lo_pct = round(float(min_val_anomaly_ratio) * 100.0, 1)
    val_hi_pct = round(float(max_val_anomaly_ratio) * 100.0, 1)
    return (
        "We built same-application cross-run hard pairs only. "
        f"Source runs were restricted to type_id in {{{src_text}}}, while target runs were restricted to disturbed type_id in {{{tgt_text}}}. "
        "Within the buildable same-app hard pool, we retained a narrow adaptation window by keeping only pairs whose descending PAD rank fell between "
        f"the Q{rank_lo_pct} and Q{rank_hi_pct} positions of the buildable pool, whose validation anomaly ratio lay between {val_lo_pct}% and {val_hi_pct}%, "
        f"whose validation anomaly count was at least {int(min_val_anomaly_count)}, and whose source train-normal support was at least {int(min_train_normal_count)} windows. "
        "This excludes anomaly-poor validation slices, extremely shifted outliers, and undersized source runs while remaining fully pre-benchmark and metadata-driven."
    )


def build_same_app_stable_adaptation_window_methods_text(
    *,
    source_type_ids,
    min_pad_rank_quantile,
    max_pad_rank_quantile,
    min_val_anomaly_ratio,
    max_val_anomaly_ratio,
    min_val_anomaly_count,
    min_train_normal_count,
):
    src_text = "all" if source_type_ids is None else ",".join(str(int(v)) for v in sorted(source_type_ids))
    rank_lo_pct = int(round(max(0.0, min(1.0, float(min_pad_rank_quantile))) * 100.0))
    rank_hi_pct = int(round(max(0.0, min(1.0, float(max_pad_rank_quantile))) * 100.0))
    val_lo_pct = round(float(min_val_anomaly_ratio) * 100.0, 1)
    val_hi_pct = round(float(max_val_anomaly_ratio) * 100.0, 1)
    return (
        "We built same-application cross-run hard pairs only. "
        f"Source runs were restricted to type_id in {{{src_text}}}, while all same-application target run types were allowed. "
        "Within the buildable same-app hard pool, we retained a stable adaptation window by keeping only pairs whose descending buildable PAD rank fell between "
        f"the Q{rank_lo_pct} and Q{rank_hi_pct} positions of the buildable pool, whose validation anomaly ratio lay between {val_lo_pct}% and {val_hi_pct}%, "
        f"whose validation anomaly count was at least {int(min_val_anomaly_count)}, and whose source train-normal support was at least {int(min_train_normal_count)} windows. "
        "This keeps moderate-to-high but non-extreme distribution shifts while excluding anomaly-poor validation slices and undersized source runs, using only pre-benchmark metadata."
    )


def compute_quantile_rank_bounds(n_items: int, min_quantile: float, max_quantile: float):
    if n_items <= 0:
        return 0, 0
    min_q = max(0.0, min(1.0, float(min_quantile)))
    max_q = max(0.0, min(1.0, float(max_quantile)))
    if max_q < min_q:
        max_q = min_q
    lo = int(math.ceil(min_q * n_items))
    hi = int(math.floor(max_q * n_items))
    lo = max(1, min(lo, n_items))
    hi = max(lo, min(hi, n_items))
    return lo, hi


USE_ARG_TARGET_TYPE_IDS = object()


def prepare_same_app_disturbed_build_rows(
    args,
    traces,
    protocol_dir: Path,
    target_type_ids_override=USE_ARG_TARGET_TYPE_IDS,
):
    source_type_ids = parse_source_type_ids(args.source_type_ids)
    if target_type_ids_override is USE_ARG_TARGET_TYPE_IDS:
        target_type_ids = parse_source_type_ids(args.same_app_disturbed_target_type_ids)
    else:
        target_type_ids = target_type_ids_override
    trace_map = {trace_dir.name: trace_dir for trace_dir in traces}
    trace_meta = {trace_dir.name: read_trace_meta(trace_dir) for trace_dir in traces}
    source_traces = [trace_dir for trace_dir in traces if is_allowed_source(trace_dir, source_type_ids)]
    if not source_traces:
        raise ValueError("No source traces left after applying source_type_ids filter.")

    candidate_rows = []
    for source_dir in source_traces:
        Xs_norm = source_norm_windows(source_dir)
        for target_dir in candidate_cross_targets(source_dir, traces, same_app_only=True):
            tgt_meta = trace_meta.get(target_dir.name, {})
            target_type_id = tgt_meta.get("type_id")
            if target_type_ids is not None and target_type_id not in target_type_ids:
                continue
            Xt, yt = target_windows(target_dir)
            target_anom = int((yt == 1).sum())
            shift = compute_domain_shift_metrics(Xs_norm, Xt, seed=args.seed)
            candidate_rows.append(
                {
                    "source_entity": source_dir.name,
                    "target_entity": target_dir.name,
                    "pair_id": f"{source_dir.name}__to__{target_dir.name}",
                    "source_app": trace_meta.get(source_dir.name, {}).get("app_name") or trace_family(source_dir.name),
                    "target_app": tgt_meta.get("app_name") or trace_family(target_dir.name),
                    "source_type_id": trace_meta.get(source_dir.name, {}).get("type_id"),
                    "target_type_id": target_type_id,
                    "pad_value": float(shift.get("feature_mean_l2", float("nan"))),
                    "pad_domain_acc": float(shift.get("domain_acc", float("nan"))),
                    "pad_domain_auc": float(shift.get("domain_auc", float("nan"))),
                    "pad_feature_mean_l2": float(shift.get("feature_mean_l2", float("nan"))),
                    "target_anomaly_windows": target_anom,
                    "candidate_pair_shift_precheck": {
                        "pad_value": float(shift.get("feature_mean_l2", float("nan"))),
                        "domain_acc": float(shift.get("domain_acc", float("nan"))),
                        "domain_auc": float(shift.get("domain_auc", float("nan"))),
                        "feature_mean_l2": float(shift.get("feature_mean_l2", float("nan"))),
                    },
                    "semantic_eligible": target_anom >= args.min_target_anom,
                    "eligibility_reason": ("ok" if target_anom >= args.min_target_anom else "target_anomaly_count_below_min"),
                }
            )

    candidate_rows.sort(
        key=lambda row: (
            float(row.get("pad_value", float("-inf"))),
            float(row.get("pad_domain_acc", float("-inf"))),
            float(row.get("pad_domain_auc", float("-inf"))),
            int(row.get("target_anomaly_windows", 0)),
        ),
        reverse=True,
    )
    for idx, row in enumerate(candidate_rows, start=1):
        row["global_pad_rank"] = idx

    build_rows = []
    build_meta_by_pair = {}
    for candidate_row in candidate_rows:
        base_row = {
            "global_pad_rank": candidate_row["global_pad_rank"],
            "source_entity": candidate_row["source_entity"],
            "target_entity": candidate_row["target_entity"],
            "pair_id": candidate_row["pair_id"],
            "shift_level": "hard",
            "source_app": candidate_row["source_app"],
            "target_app": candidate_row["target_app"],
            "source_type_id": candidate_row["source_type_id"],
            "target_type_id": candidate_row["target_type_id"],
            "pad_value": candidate_row["pad_value"],
            "pad_domain_acc": candidate_row["pad_domain_acc"],
            "pad_domain_auc": candidate_row["pad_domain_auc"],
            "build_success": False,
            "eligible": False,
            "selected": False,
            "eligibility_reason": candidate_row["eligibility_reason"],
            "failure_reason": "",
            "train_normal_count": "",
            "target_pool_count": "",
            "val_count": "",
            "test_count": "",
            "target_pool_hidden_anomaly_count": "",
            "target_pool_hidden_anomaly_ratio": "",
            "val_anomaly_count": "",
            "test_anomaly_count": "",
            "val_anomaly_ratio": "",
        }
        if not candidate_row["semantic_eligible"]:
            build_rows.append(base_row)
            continue

        source_dir = trace_map[candidate_row["source_entity"]]
        target_dir = trace_map[candidate_row["target_entity"]]
        out_dir = protocol_dir / candidate_row["pair_id"]
        ds_args = build_args(
            source_dir=source_dir,
            target_dir=target_dir,
            out_dir=out_dir,
            split_mode="search",
            shift_level="hard",
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
            seed=args.seed,
        )
        try:
            meta = create_dataset(ds_args)
            build_meta_by_pair[candidate_row["pair_id"]] = meta
            measured_pad = source_pool_pad(meta)
            build_rows.append(
                {
                    **base_row,
                    "pad_value": measured_pad["pad_value"],
                    "pad_domain_acc": measured_pad["domain_acc"],
                    "pad_domain_auc": measured_pad["domain_auc"],
                    "build_success": True,
                    "train_normal_count": meta.get("train_normal_count", ""),
                    "target_pool_count": meta.get("target_pool_count", ""),
                    "val_count": meta.get("val_count", ""),
                    "test_count": meta.get("test_count", ""),
                    "target_pool_hidden_anomaly_count": meta.get("target_pool_hidden_anomaly_count", ""),
                    "target_pool_hidden_anomaly_ratio": meta.get("target_pool_hidden_anomaly_ratio", ""),
                    "val_anomaly_count": meta.get("val_anomaly_count", ""),
                    "test_anomaly_count": meta.get("test_anomaly_count", ""),
                    "val_anomaly_ratio": (
                        (float(meta.get("val_anomaly_count", 0)) / float(meta.get("val_count", 1)))
                        if meta.get("val_count", 0)
                        else float("nan")
                    ),
                    "eligibility_reason": "ok",
                }
            )
        except Exception as exc:
            build_rows.append(
                {
                    **base_row,
                    "failure_reason": str(exc),
                    "eligibility_reason": "build_failed",
                }
            )

    return {
        "candidate_rows": candidate_rows,
        "build_rows": build_rows,
        "build_meta_by_pair": build_meta_by_pair,
        "source_type_ids": source_type_ids,
        "target_type_ids": target_type_ids,
    }


def build_same_app_disturbed_moderate_pad_pairs(args, traces, out_root: Path):
    protocol_dir = out_root / RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD
    protocol_dir.mkdir(parents=True, exist_ok=True)

    source_type_ids = parse_source_type_ids(args.source_type_ids)
    target_type_ids = parse_source_type_ids(args.same_app_disturbed_target_type_ids)
    trace_map = {trace_dir.name: trace_dir for trace_dir in traces}
    trace_meta = {trace_dir.name: read_trace_meta(trace_dir) for trace_dir in traces}
    source_traces = [trace_dir for trace_dir in traces if is_allowed_source(trace_dir, source_type_ids)]
    if not source_traces:
        raise ValueError("No source traces left after applying source_type_ids filter.")

    candidate_rows = []
    for source_dir in source_traces:
        Xs_norm = source_norm_windows(source_dir)
        for target_dir in candidate_cross_targets(source_dir, traces, same_app_only=True):
            tgt_meta = trace_meta.get(target_dir.name, {})
            target_type_id = tgt_meta.get("type_id")
            if target_type_ids is not None and target_type_id not in target_type_ids:
                continue
            Xt, yt = target_windows(target_dir)
            target_anom = int((yt == 1).sum())
            shift = compute_domain_shift_metrics(Xs_norm, Xt, seed=args.seed)
            candidate_rows.append(
                {
                    "source_entity": source_dir.name,
                    "target_entity": target_dir.name,
                    "pair_id": f"{source_dir.name}__to__{target_dir.name}",
                    "source_app": trace_meta.get(source_dir.name, {}).get("app_name") or trace_family(source_dir.name),
                    "target_app": tgt_meta.get("app_name") or trace_family(target_dir.name),
                    "source_type_id": trace_meta.get(source_dir.name, {}).get("type_id"),
                    "target_type_id": target_type_id,
                    "pad_value": float(shift.get("feature_mean_l2", float("nan"))),
                    "pad_domain_acc": float(shift.get("domain_acc", float("nan"))),
                    "pad_domain_auc": float(shift.get("domain_auc", float("nan"))),
                    "pad_feature_mean_l2": float(shift.get("feature_mean_l2", float("nan"))),
                    "target_anomaly_windows": target_anom,
                    "candidate_pair_shift_precheck": {
                        "pad_value": float(shift.get("feature_mean_l2", float("nan"))),
                        "domain_acc": float(shift.get("domain_acc", float("nan"))),
                        "domain_auc": float(shift.get("domain_auc", float("nan"))),
                        "feature_mean_l2": float(shift.get("feature_mean_l2", float("nan"))),
                    },
                    "semantic_eligible": target_anom >= args.min_target_anom,
                    "eligibility_reason": ("ok" if target_anom >= args.min_target_anom else "target_anomaly_count_below_min"),
                }
            )

    candidate_rows.sort(
        key=lambda row: (
            float(row.get("pad_value", float("-inf"))),
            float(row.get("pad_domain_acc", float("-inf"))),
            float(row.get("pad_domain_auc", float("-inf"))),
            int(row.get("target_anomaly_windows", 0)),
        ),
        reverse=True,
    )
    for idx, row in enumerate(candidate_rows, start=1):
        row["global_pad_rank"] = idx

    candidate_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "target_anomaly_windows",
        "semantic_eligible",
        "eligibility_reason",
    ]
    save_json(
        protocol_dir / "pair_candidates_ranked.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD,
            "dataset": "exathlon",
            "selected_count": 0,
            "methods_text": build_same_app_disturbed_moderate_methods_text(
                source_type_ids=source_type_ids,
                target_type_ids=target_type_ids,
                max_pad_quantile=args.same_app_disturbed_max_pad_quantile,
            ),
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": (None if target_type_ids is None else sorted(target_type_ids)),
                "same_app_only": True,
                "max_pad_quantile": float(args.same_app_disturbed_max_pad_quantile),
                "min_target_anom": int(args.min_target_anom),
            },
            "rows": public_rows(candidate_rows),
        },
    )
    write_table_csv(protocol_dir / "pair_candidates_ranked.csv", candidate_rows, candidate_columns)
    write_table_markdown(
        protocol_dir / "pair_candidates_ranked.md",
        candidate_rows,
        candidate_columns,
        title="Exathlon Same-App Disturbed Moderate-PAD Candidates",
    )

    build_rows = []
    build_meta_by_pair = {}
    for candidate_row in candidate_rows:
        base_row = {
            "global_pad_rank": candidate_row["global_pad_rank"],
            "source_entity": candidate_row["source_entity"],
            "target_entity": candidate_row["target_entity"],
            "pair_id": candidate_row["pair_id"],
            "source_app": candidate_row["source_app"],
            "target_app": candidate_row["target_app"],
            "source_type_id": candidate_row["source_type_id"],
            "target_type_id": candidate_row["target_type_id"],
            "pad_value": candidate_row["pad_value"],
            "pad_domain_acc": candidate_row["pad_domain_acc"],
            "pad_domain_auc": candidate_row["pad_domain_auc"],
            "build_success": False,
            "eligible": False,
            "selected": False,
            "eligibility_reason": candidate_row["eligibility_reason"],
            "failure_reason": "",
            "train_normal_count": "",
            "target_pool_count": "",
            "val_count": "",
            "test_count": "",
            "target_pool_hidden_anomaly_count": "",
            "target_pool_hidden_anomaly_ratio": "",
            "val_anomaly_count": "",
            "test_anomaly_count": "",
            "val_anomaly_ratio": "",
        }
        if not candidate_row["semantic_eligible"]:
            build_rows.append(base_row)
            continue

        source_dir = trace_map[candidate_row["source_entity"]]
        target_dir = trace_map[candidate_row["target_entity"]]
        out_dir = protocol_dir / candidate_row["pair_id"]
        ds_args = build_args(
            source_dir=source_dir,
            target_dir=target_dir,
            out_dir=out_dir,
            split_mode="search",
            shift_level="hard",
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
            seed=args.seed,
        )
        try:
            meta = create_dataset(ds_args)
            build_meta_by_pair[candidate_row["pair_id"]] = meta
            measured_pad = source_pool_pad(meta)
            build_rows.append(
                {
                    **base_row,
                    "pad_value": measured_pad["pad_value"],
                    "pad_domain_acc": measured_pad["domain_acc"],
                    "pad_domain_auc": measured_pad["domain_auc"],
                    "build_success": True,
                    "train_normal_count": meta.get("train_normal_count", ""),
                    "target_pool_count": meta.get("target_pool_count", ""),
                    "val_count": meta.get("val_count", ""),
                    "test_count": meta.get("test_count", ""),
                    "target_pool_hidden_anomaly_count": meta.get("target_pool_hidden_anomaly_count", ""),
                    "target_pool_hidden_anomaly_ratio": meta.get("target_pool_hidden_anomaly_ratio", ""),
                    "val_anomaly_count": meta.get("val_anomaly_count", ""),
                    "test_anomaly_count": meta.get("test_anomaly_count", ""),
                    "val_anomaly_ratio": (
                        (float(meta.get("val_anomaly_count", 0)) / float(meta.get("val_count", 1)))
                        if meta.get("val_count", 0)
                        else float("nan")
                    ),
                    "eligibility_reason": "ok",
                }
            )
        except Exception as exc:
            build_rows.append(
                {
                    **base_row,
                    "failure_reason": str(exc),
                    "eligibility_reason": "build_failed",
                }
            )

    buildable_rows = [row for row in build_rows if row.get("build_success")]
    if not buildable_rows:
        raise ValueError("No buildable same-app disturbed pairs were produced for Exathlon.")

    max_pad_value = finite_quantile(
        [row.get("pad_value") for row in buildable_rows],
        args.same_app_disturbed_max_pad_quantile,
    )
    selected_rows = []
    for row in build_rows:
        if not row.get("build_success"):
            continue
        if float(row.get("pad_value", float("nan"))) <= float(max_pad_value):
            row["eligible"] = True
            row["selected"] = True
            row["eligibility_reason"] = "ok"
            selected_rows.append(row)
        else:
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "pad_above_rule_max_quantile"

    selected_rows.sort(
        key=lambda row: (
            int(row.get("global_pad_rank", 10**9)),
            str(row.get("source_entity", "")),
            str(row.get("target_entity", "")),
        )
    )

    manifest = []
    for selection_order, row in enumerate(selected_rows, start=1):
        meta = dict(build_meta_by_pair[row["pair_id"]])
        meta["candidate_pair_shift_precheck"] = next(
            candidate["candidate_pair_shift_precheck"]
            for candidate in candidate_rows
            if candidate["pair_id"] == row["pair_id"]
        )
        meta["ranking_rule"] = RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD
        meta["global_pad_rank"] = int(row["global_pad_rank"])
        meta["selection_order"] = int(selection_order)
        meta["same_app_only"] = True
        meta["rule_target_type_ids"] = (None if target_type_ids is None else sorted(target_type_ids))
        meta["rule_max_pad_quantile"] = float(args.same_app_disturbed_max_pad_quantile)
        meta["rule_max_pad_value"] = float(max_pad_value)
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "build_success",
        "eligible",
        "selected",
        "eligibility_reason",
        "val_anomaly_ratio",
        "target_pool_hidden_anomaly_ratio",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "failure_reason",
    ]
    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD,
            "dataset": "exathlon",
            "selected_count": int(len(manifest)),
            "methods_text": build_same_app_disturbed_moderate_methods_text(
                source_type_ids=source_type_ids,
                target_type_ids=target_type_ids,
                max_pad_quantile=args.same_app_disturbed_max_pad_quantile,
            ),
            "thresholds": {
                "max_pad_value": float(max_pad_value),
            },
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": (None if target_type_ids is None else sorted(target_type_ids)),
                "same_app_only": True,
                "shift_level": "hard",
                "rule_target_pool_frac": args.target_pool_frac,
                "rule_val_frac": args.val_frac,
                "rule_guard": args.guard,
                "rule_search_step": args.search_step,
                "rule_min_target_pool": args.min_target_pool,
                "rule_min_val": args.min_val,
                "rule_min_test": args.min_test,
                "rule_min_anom_val": args.min_anom_val,
                "rule_min_anom_test": args.min_anom_test,
                "rule_min_target_anom": int(args.min_target_anom),
                "rule_max_pad_quantile": float(args.same_app_disturbed_max_pad_quantile),
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="Exathlon Same-App Disturbed Moderate-PAD Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected same-app disturbed moderate-PAD pairs: {len(manifest)}")


def build_same_app_adaptation_window_pairs(args, traces, out_root: Path):
    protocol_dir = out_root / RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW
    protocol_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_same_app_disturbed_build_rows(args, traces, protocol_dir)
    candidate_rows = bundle["candidate_rows"]
    build_rows = bundle["build_rows"]
    build_meta_by_pair = bundle["build_meta_by_pair"]
    source_type_ids = bundle["source_type_ids"]
    target_type_ids = bundle["target_type_ids"]

    candidate_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "target_anomaly_windows",
        "semantic_eligible",
        "eligibility_reason",
    ]
    methods_text = build_same_app_adaptation_window_methods_text(
        source_type_ids=source_type_ids,
        target_type_ids=target_type_ids,
        max_pad_value=args.same_app_adaptation_max_pad,
        min_val_anomaly_ratio=args.same_app_adaptation_min_val_anomaly_ratio,
        min_train_normal_count=args.same_app_adaptation_min_train_normals,
        max_source_probe_val_auroc=args.same_app_adaptation_max_source_probe_val_auroc,
    )
    save_json(
        protocol_dir / "pair_candidates_ranked.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW,
            "dataset": "exathlon",
            "selected_count": 0,
            "methods_text": methods_text,
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": (None if target_type_ids is None else sorted(target_type_ids)),
                "same_app_only": True,
                "min_target_anom": int(args.min_target_anom),
            },
            "rows": public_rows(candidate_rows),
        },
    )
    write_table_csv(protocol_dir / "pair_candidates_ranked.csv", candidate_rows, candidate_columns)
    write_table_markdown(
        protocol_dir / "pair_candidates_ranked.md",
        candidate_rows,
        candidate_columns,
        title="Exathlon Same-App Adaptation-Window Candidates",
    )

    buildable_rows = [row for row in build_rows if row.get("build_success")]
    if not buildable_rows:
        raise ValueError("No buildable same-app disturbed pairs were produced for Exathlon.")

    metadata_probe_rows = [
        row
        for row in buildable_rows
        if float(row.get("pad_value", float("inf"))) <= float(args.same_app_adaptation_max_pad)
        and float(row.get("val_anomaly_ratio", float("-inf"))) >= float(args.same_app_adaptation_min_val_anomaly_ratio)
        and int(row.get("train_normal_count", 0) or 0) >= int(args.same_app_adaptation_min_train_normals)
    ]

    print(f"[INFO] Running source-only probes for {len(metadata_probe_rows)} metadata-feasible Exathlon pairs...")
    for idx, row in enumerate(metadata_probe_rows, start=1):
        print(f"[PROBE {idx}/{len(metadata_probe_rows)}] {row['pair_id']}")
        probe = run_source_probe_for_row(protocol_dir, row, args)
        row["probe_success"] = bool(probe.get("probe_success"))
        row["probe_failure_reason"] = probe.get("failure_reason", "")
        row["source_probe_best_arch"] = probe.get("best_arch")
        row["source_probe_val_auroc"] = float(probe.get("source_probe_val_auroc", float("nan")))
        row["source_probe_val_auprc"] = float(probe.get("source_probe_val_auprc", float("nan")))
        row["source_probe_val_f1_best"] = float(probe.get("source_probe_val_f1_best", float("nan")))
        row["source_probe_log_path"] = probe.get("log_path", "")

    selected_rows = []
    for row in build_rows:
        row.setdefault("probe_success", False)
        row.setdefault("probe_failure_reason", "")
        row.setdefault("source_probe_best_arch", "")
        row.setdefault("source_probe_val_auroc", float("nan"))
        row.setdefault("source_probe_val_auprc", float("nan"))
        row.setdefault("source_probe_val_f1_best", float("nan"))
        row.setdefault("source_probe_log_path", "")

        if not row.get("build_success"):
            continue
        if float(row.get("pad_value", float("inf"))) > float(args.same_app_adaptation_max_pad):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "pad_above_rule_max"
            continue
        if float(row.get("val_anomaly_ratio", float("-inf"))) < float(args.same_app_adaptation_min_val_anomaly_ratio):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "val_anomaly_ratio_below_rule_min"
            continue
        if int(row.get("train_normal_count", 0) or 0) < int(args.same_app_adaptation_min_train_normals):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "train_normal_count_below_rule_min"
            continue
        if not row.get("probe_success"):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "probe_failed"
            continue
        if float(row.get("source_probe_val_auroc", float("inf"))) >= float(args.same_app_adaptation_max_source_probe_val_auroc):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "source_probe_val_auroc_above_rule_max"
            continue

        row["eligible"] = True
        row["selected"] = True
        row["eligibility_reason"] = "ok"
        selected_rows.append(row)

    selected_rows.sort(
        key=lambda row: (
            int(row.get("global_pad_rank", 10**9)),
            str(row.get("source_entity", "")),
            str(row.get("target_entity", "")),
        )
    )

    candidate_shift_by_pair = {
        row["pair_id"]: row["candidate_pair_shift_precheck"] for row in candidate_rows
    }
    manifest = []
    for selection_order, row in enumerate(selected_rows, start=1):
        meta = dict(build_meta_by_pair[row["pair_id"]])
        meta["candidate_pair_shift_precheck"] = candidate_shift_by_pair[row["pair_id"]]
        meta["ranking_rule"] = RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW
        meta["global_pad_rank"] = int(row["global_pad_rank"])
        meta["selection_order"] = int(selection_order)
        meta["same_app_only"] = True
        meta["rule_target_type_ids"] = (None if target_type_ids is None else sorted(target_type_ids))
        meta["same_app_adaptation_window"] = {
            "source_probe_best_arch": row["source_probe_best_arch"],
            "source_probe_val_auroc": float(row["source_probe_val_auroc"]),
            "source_probe_val_auprc": float(row["source_probe_val_auprc"]),
            "source_probe_val_f1_best": float(row["source_probe_val_f1_best"]),
            "max_pad_value": float(args.same_app_adaptation_max_pad),
            "min_val_anomaly_ratio": float(args.same_app_adaptation_min_val_anomaly_ratio),
            "min_train_normal_count": int(args.same_app_adaptation_min_train_normals),
            "max_source_probe_val_auroc": float(args.same_app_adaptation_max_source_probe_val_auroc),
            "rule_text": methods_text,
        }
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "shift_level",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "build_success",
        "probe_success",
        "eligible",
        "selected",
        "eligibility_reason",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "val_anomaly_ratio",
        "target_pool_hidden_anomaly_ratio",
        "source_probe_best_arch",
        "source_probe_val_auroc",
        "source_probe_val_auprc",
        "source_probe_val_f1_best",
        "probe_failure_reason",
        "failure_reason",
    ]
    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW,
            "dataset": "exathlon",
            "selected_count": int(len(manifest)),
            "methods_text": methods_text,
            "thresholds": {
                "max_pad_value": float(args.same_app_adaptation_max_pad),
                "min_val_anomaly_ratio": float(args.same_app_adaptation_min_val_anomaly_ratio),
                "min_train_normal_count": int(args.same_app_adaptation_min_train_normals),
                "max_source_probe_val_auroc": float(args.same_app_adaptation_max_source_probe_val_auroc),
            },
            "probe_config": {
                "epochs_pretrain": int(args.probe_epochs_pretrain),
                "search_candidates": int(args.probe_search_candidates),
                "nas_search_iters": int(args.probe_nas_search_iters),
                "nas_search_strategy": str(args.probe_nas_search_strategy),
                "batch_size": int(args.probe_batch_size),
                "device": str(args.probe_device),
                "oneclass_method": str(args.probe_oneclass_method),
                "oneclass_epochs": int(args.probe_oneclass_epochs),
                "oneclass_final_epochs": int(args.probe_oneclass_final_epochs),
                "oneclass_batch_size": int(args.probe_oneclass_batch_size),
                "oneclass_max_fit": int(args.probe_oneclass_max_fit),
            },
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": (None if target_type_ids is None else sorted(target_type_ids)),
                "same_app_only": True,
                "shift_level": "hard",
                "rule_target_pool_frac": args.target_pool_frac,
                "rule_val_frac": args.val_frac,
                "rule_guard": args.guard,
                "rule_search_step": args.search_step,
                "rule_min_target_pool": args.min_target_pool,
                "rule_min_val": args.min_val,
                "rule_min_test": args.min_test,
                "rule_min_anom_val": args.min_anom_val,
                "rule_min_anom_test": args.min_anom_test,
                "rule_min_target_anom": int(args.min_target_anom),
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="Exathlon Same-App Adaptation-Window Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected same-app adaptation-window pairs: {len(manifest)}")


def build_same_app_stable_adaptation_window_pairs(args, traces, out_root: Path):
    protocol_dir = out_root / RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW
    protocol_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_same_app_disturbed_build_rows(
        args,
        traces,
        protocol_dir,
        target_type_ids_override=None,
    )
    candidate_rows = bundle["candidate_rows"]
    build_rows = bundle["build_rows"]
    build_meta_by_pair = bundle["build_meta_by_pair"]
    source_type_ids = bundle["source_type_ids"]

    methods_text = build_same_app_stable_adaptation_window_methods_text(
        source_type_ids=source_type_ids,
        min_pad_rank_quantile=args.same_app_stable_min_pad_rank_quantile,
        max_pad_rank_quantile=args.same_app_stable_max_pad_rank_quantile,
        min_val_anomaly_ratio=args.same_app_stable_min_val_anomaly_ratio,
        max_val_anomaly_ratio=args.same_app_stable_max_val_anomaly_ratio,
        min_val_anomaly_count=args.same_app_stable_min_val_anomalies,
        min_train_normal_count=args.same_app_stable_min_train_normals,
    )
    candidate_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "target_anomaly_windows",
        "semantic_eligible",
        "eligibility_reason",
    ]
    save_json(
        protocol_dir / "pair_candidates_ranked.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW,
            "dataset": "exathlon",
            "selected_count": 0,
            "methods_text": methods_text,
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": None,
                "same_app_only": True,
                "min_target_anom": int(args.min_target_anom),
            },
            "rows": public_rows(candidate_rows),
        },
    )
    write_table_csv(protocol_dir / "pair_candidates_ranked.csv", candidate_rows, candidate_columns)
    write_table_markdown(
        protocol_dir / "pair_candidates_ranked.md",
        candidate_rows,
        candidate_columns,
        title="Exathlon Same-App Stable Adaptation-Window Candidates",
    )

    buildable_rows = [row for row in build_rows if row.get("build_success")]
    if not buildable_rows:
        raise ValueError("No buildable same-app hard pairs were produced for Exathlon stable adaptation-window rule.")

    buildable_rows.sort(
        key=lambda row: (
            int(row.get("global_pad_rank", 10**9)),
            str(row.get("source_entity", "")),
            str(row.get("target_entity", "")),
        )
    )
    for idx, row in enumerate(buildable_rows, start=1):
        row["buildable_pad_rank"] = int(idx)

    min_buildable_pad_rank, max_buildable_pad_rank = compute_quantile_rank_bounds(
        len(buildable_rows),
        args.same_app_stable_min_pad_rank_quantile,
        args.same_app_stable_max_pad_rank_quantile,
    )

    selected_rows = []
    for row in build_rows:
        row.setdefault("buildable_pad_rank", "")
        if not row.get("build_success"):
            continue
        if not (
            min_buildable_pad_rank
            <= int(row.get("buildable_pad_rank", 0) or 0)
            <= max_buildable_pad_rank
        ):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "buildable_pad_rank_outside_rule_band"
            continue
        val_ratio = float(row.get("val_anomaly_ratio", float("nan")))
        if not (
            float(args.same_app_stable_min_val_anomaly_ratio)
            <= val_ratio
            <= float(args.same_app_stable_max_val_anomaly_ratio)
        ):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "val_anomaly_ratio_outside_rule_window"
            continue
        if int(row.get("val_anomaly_count", 0) or 0) < int(args.same_app_stable_min_val_anomalies):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "val_anomaly_count_below_rule_min"
            continue
        if int(row.get("train_normal_count", 0) or 0) < int(args.same_app_stable_min_train_normals):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "train_normal_count_below_rule_min"
            continue

        row["eligible"] = True
        row["selected"] = True
        row["eligibility_reason"] = "ok"
        selected_rows.append(row)

    selected_rows.sort(
        key=lambda row: (
            int(row.get("buildable_pad_rank", 10**9) or 10**9),
            int(row.get("global_pad_rank", 10**9)),
            str(row.get("source_entity", "")),
            str(row.get("target_entity", "")),
        )
    )

    candidate_shift_by_pair = {
        row["pair_id"]: row["candidate_pair_shift_precheck"] for row in candidate_rows
    }
    manifest = []
    for selection_order, row in enumerate(selected_rows, start=1):
        meta = dict(build_meta_by_pair[row["pair_id"]])
        meta["candidate_pair_shift_precheck"] = candidate_shift_by_pair[row["pair_id"]]
        meta["ranking_rule"] = RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW
        meta["global_pad_rank"] = int(row["global_pad_rank"])
        meta["buildable_pad_rank"] = int(row["buildable_pad_rank"])
        meta["selection_order"] = int(selection_order)
        meta["same_app_only"] = True
        meta["rule_target_type_ids"] = None
        meta["same_app_stable_adaptation_window"] = {
            "min_buildable_pad_rank_quantile": float(args.same_app_stable_min_pad_rank_quantile),
            "max_buildable_pad_rank_quantile": float(args.same_app_stable_max_pad_rank_quantile),
            "min_buildable_pad_rank": int(min_buildable_pad_rank),
            "max_buildable_pad_rank": int(max_buildable_pad_rank),
            "min_val_anomaly_ratio": float(args.same_app_stable_min_val_anomaly_ratio),
            "max_val_anomaly_ratio": float(args.same_app_stable_max_val_anomaly_ratio),
            "min_val_anomaly_count": int(args.same_app_stable_min_val_anomalies),
            "min_train_normal_count": int(args.same_app_stable_min_train_normals),
            "rule_text": methods_text,
        }
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "buildable_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "shift_level",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "build_success",
        "eligible",
        "selected",
        "eligibility_reason",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "val_anomaly_ratio",
        "target_pool_hidden_anomaly_ratio",
        "failure_reason",
    ]
    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW,
            "dataset": "exathlon",
            "selected_count": int(len(manifest)),
            "methods_text": methods_text,
            "thresholds": {
                "min_buildable_pad_rank_quantile": float(args.same_app_stable_min_pad_rank_quantile),
                "max_buildable_pad_rank_quantile": float(args.same_app_stable_max_pad_rank_quantile),
                "min_buildable_pad_rank": int(min_buildable_pad_rank),
                "max_buildable_pad_rank": int(max_buildable_pad_rank),
                "min_val_anomaly_ratio": float(args.same_app_stable_min_val_anomaly_ratio),
                "max_val_anomaly_ratio": float(args.same_app_stable_max_val_anomaly_ratio),
                "min_val_anomaly_count": int(args.same_app_stable_min_val_anomalies),
                "min_train_normal_count": int(args.same_app_stable_min_train_normals),
            },
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": None,
                "same_app_only": True,
                "shift_level": "hard",
                "rule_target_pool_frac": args.target_pool_frac,
                "rule_val_frac": args.val_frac,
                "rule_guard": args.guard,
                "rule_search_step": args.search_step,
                "rule_min_target_pool": args.min_target_pool,
                "rule_min_val": args.min_val,
                "rule_min_test": args.min_test,
                "rule_min_anom_val": args.min_anom_val,
                "rule_min_anom_test": args.min_anom_test,
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="Exathlon Same-App Stable Adaptation-Window Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected same-app stable adaptation-window pairs: {len(manifest)}")


def build_same_app_narrow_adaptation_window_pairs(args, traces, out_root: Path):
    protocol_dir = out_root / RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW
    protocol_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_same_app_disturbed_build_rows(args, traces, protocol_dir)
    candidate_rows = bundle["candidate_rows"]
    build_rows = bundle["build_rows"]
    build_meta_by_pair = bundle["build_meta_by_pair"]
    source_type_ids = bundle["source_type_ids"]
    target_type_ids = bundle["target_type_ids"]

    methods_text = build_same_app_narrow_adaptation_window_methods_text(
        source_type_ids=source_type_ids,
        target_type_ids=target_type_ids,
        min_pad_rank_quantile=args.same_app_narrow_min_pad_rank_quantile,
        max_pad_rank_quantile=args.same_app_narrow_max_pad_rank_quantile,
        min_val_anomaly_ratio=args.same_app_narrow_min_val_anomaly_ratio,
        max_val_anomaly_ratio=args.same_app_narrow_max_val_anomaly_ratio,
        min_val_anomaly_count=args.same_app_narrow_min_val_anomalies,
        min_train_normal_count=args.same_app_narrow_min_train_normals,
    )
    candidate_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "target_anomaly_windows",
        "semantic_eligible",
        "eligibility_reason",
    ]
    save_json(
        protocol_dir / "pair_candidates_ranked.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW,
            "dataset": "exathlon",
            "selected_count": 0,
            "methods_text": methods_text,
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": (None if target_type_ids is None else sorted(target_type_ids)),
                "same_app_only": True,
                "min_target_anom": int(args.min_target_anom),
            },
            "rows": public_rows(candidate_rows),
        },
    )
    write_table_csv(protocol_dir / "pair_candidates_ranked.csv", candidate_rows, candidate_columns)
    write_table_markdown(
        protocol_dir / "pair_candidates_ranked.md",
        candidate_rows,
        candidate_columns,
        title="Exathlon Same-App Narrow Adaptation-Window Candidates",
    )

    buildable_rows = [row for row in build_rows if row.get("build_success")]
    if not buildable_rows:
        raise ValueError("No buildable same-app disturbed pairs were produced for Exathlon.")

    buildable_rows.sort(
        key=lambda row: (
            int(row.get("global_pad_rank", 10**9)),
            str(row.get("source_entity", "")),
            str(row.get("target_entity", "")),
        )
    )
    for idx, row in enumerate(buildable_rows, start=1):
        row["buildable_pad_rank"] = int(idx)

    min_buildable_pad_rank, max_buildable_pad_rank = compute_quantile_rank_bounds(
        len(buildable_rows),
        args.same_app_narrow_min_pad_rank_quantile,
        args.same_app_narrow_max_pad_rank_quantile,
    )

    selected_rows = []
    for row in build_rows:
        row.setdefault("buildable_pad_rank", "")
        if not row.get("build_success"):
            continue
        if not (
            min_buildable_pad_rank
            <= int(row.get("buildable_pad_rank", 0) or 0)
            <= max_buildable_pad_rank
        ):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "buildable_pad_rank_outside_rule_band"
            continue
        val_ratio = float(row.get("val_anomaly_ratio", float("nan")))
        if not (
            float(args.same_app_narrow_min_val_anomaly_ratio)
            <= val_ratio
            <= float(args.same_app_narrow_max_val_anomaly_ratio)
        ):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "val_anomaly_ratio_outside_rule_window"
            continue
        if int(row.get("val_anomaly_count", 0) or 0) < int(args.same_app_narrow_min_val_anomalies):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "val_anomaly_count_below_rule_min"
            continue
        if int(row.get("train_normal_count", 0) or 0) < int(args.same_app_narrow_min_train_normals):
            row["eligible"] = False
            row["selected"] = False
            row["eligibility_reason"] = "train_normal_count_below_rule_min"
            continue

        row["eligible"] = True
        row["selected"] = True
        row["eligibility_reason"] = "ok"
        selected_rows.append(row)

    selected_rows.sort(
        key=lambda row: (
            int(row.get("buildable_pad_rank", 10**9) or 10**9),
            int(row.get("global_pad_rank", 10**9)),
            str(row.get("source_entity", "")),
            str(row.get("target_entity", "")),
        )
    )

    candidate_shift_by_pair = {
        row["pair_id"]: row["candidate_pair_shift_precheck"] for row in candidate_rows
    }
    manifest = []
    for selection_order, row in enumerate(selected_rows, start=1):
        meta = dict(build_meta_by_pair[row["pair_id"]])
        meta["candidate_pair_shift_precheck"] = candidate_shift_by_pair[row["pair_id"]]
        meta["ranking_rule"] = RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW
        meta["global_pad_rank"] = int(row["global_pad_rank"])
        meta["buildable_pad_rank"] = int(row["buildable_pad_rank"])
        meta["selection_order"] = int(selection_order)
        meta["same_app_only"] = True
        meta["rule_target_type_ids"] = (None if target_type_ids is None else sorted(target_type_ids))
        meta["same_app_narrow_adaptation_window"] = {
            "min_buildable_pad_rank_quantile": float(args.same_app_narrow_min_pad_rank_quantile),
            "max_buildable_pad_rank_quantile": float(args.same_app_narrow_max_pad_rank_quantile),
            "min_buildable_pad_rank": int(min_buildable_pad_rank),
            "max_buildable_pad_rank": int(max_buildable_pad_rank),
            "min_val_anomaly_ratio": float(args.same_app_narrow_min_val_anomaly_ratio),
            "max_val_anomaly_ratio": float(args.same_app_narrow_max_val_anomaly_ratio),
            "min_val_anomaly_count": int(args.same_app_narrow_min_val_anomalies),
            "min_train_normal_count": int(args.same_app_narrow_min_train_normals),
            "rule_text": methods_text,
        }
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "buildable_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "shift_level",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "build_success",
        "eligible",
        "selected",
        "eligibility_reason",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "val_anomaly_ratio",
        "target_pool_hidden_anomaly_ratio",
        "failure_reason",
    ]
    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW,
            "dataset": "exathlon",
            "selected_count": int(len(manifest)),
            "methods_text": methods_text,
            "thresholds": {
                "min_buildable_pad_rank_quantile": float(args.same_app_narrow_min_pad_rank_quantile),
                "max_buildable_pad_rank_quantile": float(args.same_app_narrow_max_pad_rank_quantile),
                "min_buildable_pad_rank": int(min_buildable_pad_rank),
                "max_buildable_pad_rank": int(max_buildable_pad_rank),
                "min_val_anomaly_ratio": float(args.same_app_narrow_min_val_anomaly_ratio),
                "max_val_anomaly_ratio": float(args.same_app_narrow_max_val_anomaly_ratio),
                "min_val_anomaly_count": int(args.same_app_narrow_min_val_anomalies),
                "min_train_normal_count": int(args.same_app_narrow_min_train_normals),
            },
            "config": {
                "source_type_ids": (None if source_type_ids is None else sorted(source_type_ids)),
                "target_type_ids": (None if target_type_ids is None else sorted(target_type_ids)),
                "same_app_only": True,
                "shift_level": "hard",
                "rule_target_pool_frac": args.target_pool_frac,
                "rule_val_frac": args.val_frac,
                "rule_guard": args.guard,
                "rule_search_step": args.search_step,
                "rule_min_target_pool": args.min_target_pool,
                "rule_min_val": args.min_val,
                "rule_min_test": args.min_test,
                "rule_min_anom_val": args.min_anom_val,
                "rule_min_anom_test": args.min_anom_test,
                "rule_min_target_anom": int(args.min_target_anom),
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="Exathlon Same-App Narrow Adaptation-Window Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected same-app narrow adaptation-window pairs: {len(manifest)}")


def build_global_top_pad_pairs(args, traces, out_root: Path):
    if not args.rankings_json:
        raise ValueError("--rankings_json is required when --pair_rule is cross_entity_hard_topPAD_global_top3")

    rankings_path = Path(args.rankings_json)
    if not rankings_path.exists():
        raise FileNotFoundError(rankings_path)

    trace_map = {trace_dir.name: trace_dir for trace_dir in traces}
    trace_meta = {trace_dir.name: read_trace_meta(trace_dir) for trace_dir in traces}
    source_type_ids = parse_source_type_ids(args.source_type_ids)
    protocol_dir = out_root / RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3
    protocol_dir.mkdir(parents=True, exist_ok=True)

    ranking_payload, ranked_rows = load_pad_pair_rankings(rankings_path)
    candidate_rows = []
    eligible_rows = []

    for row in ranked_rows:
        src_name = row["source_entity"]
        tgt_name = row["target_entity"]
        src_meta = trace_meta.get(src_name, {})
        tgt_meta = trace_meta.get(tgt_name, {})
        src_type_id = src_meta.get("type_id")
        tgt_type_id = tgt_meta.get("type_id")
        src_app = src_meta.get("app_name") or trace_family(src_name)
        tgt_app = tgt_meta.get("app_name") or trace_family(tgt_name)

        eligibility_reason = "ok"
        eligible = True
        if src_name == tgt_name:
            eligible = False
            eligibility_reason = "same_entity"
        elif src_name not in trace_map or tgt_name not in trace_map:
            eligible = False
            eligibility_reason = "filtered_out_by_entity_subset"
        elif source_type_ids is not None and src_type_id not in source_type_ids:
            eligible = False
            eligibility_reason = "source_type_id_not_allowed"

        candidate_row = {
            "global_pad_rank": row["global_pad_rank"],
            "source_entity": src_name,
            "target_entity": tgt_name,
            "pair_id": row["pair_id"],
            "pad_value": row["pad_value"],
            "pad_domain_acc": row["pad_domain_acc"],
            "pad_domain_auc": row["pad_domain_auc"],
            "pad_feature_mean_l2": row["pad_feature_mean_l2"],
            "n_source_normal_windows": row["n_source_normal_windows"],
            "n_target_normal_windows": row["n_target_normal_windows"],
            "source_app": src_app,
            "target_app": tgt_app,
            "source_type_id": src_type_id,
            "target_type_id": tgt_type_id,
            "eligible": eligible,
            "eligibility_reason": eligibility_reason,
        }
        candidate_rows.append(candidate_row)
        if eligible:
            eligible_rows.append((candidate_row, row))

    candidate_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "pad_feature_mean_l2",
        "n_source_normal_windows",
        "n_target_normal_windows",
        "source_app",
        "target_app",
        "source_type_id",
        "target_type_id",
        "eligible",
        "eligibility_reason",
    ]
    save_json(
        protocol_dir / "pair_candidates_ranked.json",
        {
            "rule_name": RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3,
            "dataset": "exathlon",
            "ranking_source": str(rankings_path),
            "ranking_notes": ranking_payload.get("notes", {}),
            "requested_topk": int(args.global_topk),
            "rows": public_rows(candidate_rows),
        },
    )
    write_table_csv(protocol_dir / "pair_candidates_ranked.csv", candidate_rows, candidate_columns)
    write_table_markdown(
        protocol_dir / "pair_candidates_ranked.md",
        candidate_rows,
        candidate_columns,
        title="Exathlon Pair Candidates",
    )

    manifest = []
    build_attempt_rows = []
    attempt_order = 0
    for candidate_row, ranking_row in eligible_rows:
        if len(manifest) >= args.global_topk:
            break
        attempt_order += 1
        source_dir = trace_map[candidate_row["source_entity"]]
        target_dir = trace_map[candidate_row["target_entity"]]
        out_dir = protocol_dir / candidate_row["pair_id"]
        ds_args = build_args(
            source_dir=source_dir,
            target_dir=target_dir,
            out_dir=out_dir,
            split_mode="search",
            shift_level="hard",
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
            seed=args.seed,
        )
        try:
            meta = create_dataset(ds_args)
            meta["candidate_pair_shift_precheck"] = ranking_row["raw_row"].get("pad_latent", {})
            meta["ranking_rule"] = RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3
            meta["ranking_source"] = str(rankings_path)
            meta["global_pad_rank"] = int(candidate_row["global_pad_rank"])
            meta["selection_order"] = int(len(manifest) + 1)
            manifest.append(meta)
            build_attempt_rows.append(
                {
                    "attempt_order": attempt_order,
                    "global_pad_rank": candidate_row["global_pad_rank"],
                    "source_entity": candidate_row["source_entity"],
                    "target_entity": candidate_row["target_entity"],
                    "pair_id": candidate_row["pair_id"],
                    "pad_value": candidate_row["pad_value"],
                    "build_success": True,
                    "selected": True,
                    "failure_reason": "",
                    "selection_order": len(manifest),
                    "train_normal_count": meta["train_normal_count"],
                    "target_pool_count": meta["target_pool_count"],
                    "val_count": meta["val_count"],
                    "test_count": meta["test_count"],
                    "target_pool_hidden_anomaly_count": meta["target_pool_hidden_anomaly_count"],
                    "val_anomaly_count": meta["val_anomaly_count"],
                    "test_anomaly_count": meta["test_anomaly_count"],
                }
            )
        except Exception as exc:
            build_attempt_rows.append(
                {
                    "attempt_order": attempt_order,
                    "global_pad_rank": candidate_row["global_pad_rank"],
                    "source_entity": candidate_row["source_entity"],
                    "target_entity": candidate_row["target_entity"],
                    "pair_id": candidate_row["pair_id"],
                    "pad_value": candidate_row["pad_value"],
                    "build_success": False,
                    "selected": False,
                    "failure_reason": str(exc),
                    "selection_order": "",
                    "train_normal_count": "",
                    "target_pool_count": "",
                    "val_count": "",
                    "test_count": "",
                    "target_pool_hidden_anomaly_count": "",
                    "val_anomaly_count": "",
                    "test_anomaly_count": "",
                }
            )

    build_columns = [
        "attempt_order",
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "pad_value",
        "build_success",
        "selected",
        "failure_reason",
        "selection_order",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "target_pool_hidden_anomaly_count",
        "val_anomaly_count",
        "test_anomaly_count",
    ]
    save_json(
        protocol_dir / "pair_build_attempts.json",
        {
            "rule_name": RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3,
            "dataset": "exathlon",
            "requested_topk": int(args.global_topk),
            "selected_count": int(len(manifest)),
            "rows": public_rows(build_attempt_rows),
        },
    )
    write_table_csv(protocol_dir / "pair_build_attempts.csv", build_attempt_rows, build_columns)
    write_table_markdown(
        protocol_dir / "pair_build_attempts.md",
        build_attempt_rows,
        build_columns,
        title="Exathlon Pair Build Attempts",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected buildable pairs: {len(manifest)} / {args.global_topk}")
    if len(manifest) < args.global_topk:
        print("[WARN] Fewer buildable hard pairs than requested top-k.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/exathlon")
    ap.add_argument("--out_root", default="data/exathlon_experiments")
    ap.add_argument("--entities", default=None, help="Comma-separated cached entity names to include.")
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--shift_levels", default="auto,hard")
    ap.add_argument("--build_temporal", action="store_true")
    ap.add_argument("--build_cross_entity", action="store_true")
    ap.add_argument(
        "--allow_cross_app",
        action="store_true",
        help="If set, allow cross-application target traces. By default, Exathlon experiments stay within the same app family.",
    )
    ap.add_argument(
        "--source_type_ids",
        default="0",
        help="Comma-separated trace type_ids allowed to act as source entities. Default '0' keeps undisturbed traces only. Use 'all' to disable filtering.",
    )
    ap.add_argument("--topk_cross", type=int, default=1)
    ap.add_argument("--target_pool_frac", type=float, default=DEFAULT_TARGET_POOL_FRAC)
    ap.add_argument("--val_frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--guard", type=int, default=DEFAULT_GUARD)
    ap.add_argument("--search_step", type=int, default=DEFAULT_SEARCH_STEP)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=DEFAULT_MAX_POOL_ANOM_RATIO)
    ap.add_argument("--min_target_pool", type=int, default=DEFAULT_MIN_TARGET_POOL)
    ap.add_argument("--min_val", type=int, default=DEFAULT_MIN_VAL)
    ap.add_argument("--min_test", type=int, default=DEFAULT_MIN_TEST)
    ap.add_argument("--min_anom_val", type=int, default=DEFAULT_MIN_ANOM_VAL)
    ap.add_argument("--min_anom_test", type=int, default=DEFAULT_MIN_ANOM_TEST)
    ap.add_argument("--min_target_anom", type=int, default=3)
    ap.add_argument(
        "--pair_rule",
        default="per_source_topk",
        choices=[
            "per_source_topk",
            RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3,
            RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD,
            RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW,
            RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW,
            RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW,
        ],
    )
    ap.add_argument("--rankings_json", default=None)
    ap.add_argument("--global_topk", type=int, default=3)
    ap.add_argument(
        "--same_app_disturbed_target_type_ids",
        default="3,4",
        help="Comma-separated target type_ids kept by the same-app disturbed moderate-PAD rule. Default '3,4'. Use 'all' to disable filtering.",
    )
    ap.add_argument(
        "--same_app_disturbed_max_pad_quantile",
        type=float,
        default=0.40,
        help="Within the buildable disturbed same-app subset, keep only pairs whose PAD is at or below this quantile. Default 0.40 keeps the lower 40%% PAD and excludes the extreme upper tail.",
    )
    ap.add_argument(
        "--same_app_adaptation_max_pad",
        type=float,
        default=150.0,
        help="Maximum PAD retained by the same-app adaptation-window rule. Default 150.0.",
    )
    ap.add_argument(
        "--same_app_adaptation_min_val_anomaly_ratio",
        type=float,
        default=0.25,
        help="Minimum val anomaly ratio retained by the same-app adaptation-window rule. Default 0.25.",
    )
    ap.add_argument(
        "--same_app_adaptation_min_train_normals",
        type=int,
        default=50,
        help="Minimum train_normal_count retained by the same-app adaptation-window rule. Default 50.",
    )
    ap.add_argument(
        "--same_app_adaptation_max_source_probe_val_auroc",
        type=float,
        default=0.99,
        help="Maximum source-only probe val AUROC retained by the same-app adaptation-window rule. Default 0.99.",
    )
    ap.add_argument(
        "--same_app_adaptation_probe_root",
        default=None,
        help="Optional directory for caching source-probe runs used by the same-app adaptation-window rule.",
    )
    ap.add_argument(
        "--same_app_adaptation_probe_force",
        action="store_true",
        help="Force rerunning source-only probes for the same-app adaptation-window rule.",
    )
    ap.add_argument(
        "--same_app_narrow_min_pad_rank_quantile",
        type=float,
        default=0.50,
        help="Lower quantile of the buildable descending PAD rank kept by the same-app narrow adaptation-window rule. Default 0.50.",
    )
    ap.add_argument(
        "--same_app_narrow_max_pad_rank_quantile",
        type=float,
        default=0.85,
        help="Upper quantile of the buildable descending PAD rank kept by the same-app narrow adaptation-window rule. Default 0.85.",
    )
    ap.add_argument(
        "--same_app_narrow_min_val_anomaly_ratio",
        type=float,
        default=0.24,
        help="Minimum val anomaly ratio kept by the same-app narrow adaptation-window rule. Default 0.24.",
    )
    ap.add_argument(
        "--same_app_narrow_max_val_anomaly_ratio",
        type=float,
        default=0.30,
        help="Maximum val anomaly ratio kept by the same-app narrow adaptation-window rule. Default 0.30.",
    )
    ap.add_argument(
        "--same_app_narrow_min_val_anomalies",
        type=int,
        default=14,
        help="Minimum val anomaly count kept by the same-app narrow adaptation-window rule. Default 14.",
    )
    ap.add_argument(
        "--same_app_narrow_min_train_normals",
        type=int,
        default=50,
        help="Minimum train_normal_count kept by the same-app narrow adaptation-window rule. Default 50.",
    )
    ap.add_argument(
        "--same_app_stable_min_pad_rank_quantile",
        type=float,
        default=0.40,
        help="Lower quantile of the buildable descending PAD rank kept by the same-app stable adaptation-window rule. Default 0.40.",
    )
    ap.add_argument(
        "--same_app_stable_max_pad_rank_quantile",
        type=float,
        default=0.90,
        help="Upper quantile of the buildable descending PAD rank kept by the same-app stable adaptation-window rule. Default 0.90.",
    )
    ap.add_argument(
        "--same_app_stable_min_val_anomaly_ratio",
        type=float,
        default=0.24,
        help="Minimum val anomaly ratio kept by the same-app stable adaptation-window rule. Default 0.24.",
    )
    ap.add_argument(
        "--same_app_stable_max_val_anomaly_ratio",
        type=float,
        default=0.30,
        help="Maximum val anomaly ratio kept by the same-app stable adaptation-window rule. Default 0.30.",
    )
    ap.add_argument(
        "--same_app_stable_min_val_anomalies",
        type=int,
        default=14,
        help="Minimum val anomaly count kept by the same-app stable adaptation-window rule. Default 14.",
    )
    ap.add_argument(
        "--same_app_stable_min_train_normals",
        type=int,
        default=50,
        help="Minimum train_normal_count kept by the same-app stable adaptation-window rule. Default 50.",
    )
    ap.add_argument("--probe_epochs_pretrain", type=int, default=6)
    ap.add_argument("--probe_search_candidates", type=int, default=5)
    ap.add_argument("--probe_nas_search_iters", type=int, default=2)
    ap.add_argument("--probe_nas_search_strategy", default="random")
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
    ap.add_argument("--probe_oneclass_batch_size", type=int, default=1024)
    ap.add_argument("--probe_oneclass_max_fit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    shift_levels = [s.strip() for s in args.shift_levels.split(",") if s.strip()]
    traces = list_trace_dirs(data_root)
    if not traces:
        raise FileNotFoundError(f"No cached Exathlon entity folders with source.npz/target.npz under {data_root}")

    if args.entities:
        keep = {e.strip() for e in args.entities.split(",") if e.strip()}
        traces = [t for t in traces if t.name in keep]
    if args.max_entities > 0:
        traces = traces[: args.max_entities]
    if not traces:
        raise ValueError("No traces left after filtering.")

    if args.pair_rule == RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3:
        build_global_top_pad_pairs(args, traces, out_root)
        return
    if args.pair_rule == RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD:
        build_same_app_disturbed_moderate_pad_pairs(args, traces, out_root)
        return
    if args.pair_rule == RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW:
        build_same_app_adaptation_window_pairs(args, traces, out_root)
        return
    if args.pair_rule == RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW:
        build_same_app_narrow_adaptation_window_pairs(args, traces, out_root)
        return
    if args.pair_rule == RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW:
        build_same_app_stable_adaptation_window_pairs(args, traces, out_root)
        return

    source_type_ids = parse_source_type_ids(args.source_type_ids)
    source_traces = [t for t in traces if is_allowed_source(t, source_type_ids)]
    if not source_traces:
        raise ValueError("No source traces left after applying source_type_ids filter.")

    build_temporal = args.build_temporal or (not args.build_temporal and not args.build_cross_entity)
    build_cross = args.build_cross_entity or (not args.build_temporal and not args.build_cross_entity)
    same_app_only = not args.allow_cross_app

    manifest = []

    if build_temporal:
        for shift_level in shift_levels:
            for source_dir in source_traces:
                out_dir = out_root / f"temporal_{shift_level}" / source_dir.name
                ds_args = build_args(
                    source_dir=source_dir,
                    target_dir=None,
                    out_dir=out_dir,
                    split_mode="search",
                    shift_level=shift_level,
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
                    seed=args.seed,
                )
                try:
                    meta = create_dataset(ds_args)
                    manifest.append(meta)
                except Exception as exc:
                    print(f"[WARN] temporal {shift_level} {source_dir.name}: {exc}")

    if build_cross:
        for shift_level in shift_levels:
            for source_dir in source_traces:
                ranked = rank_cross_targets(
                    source_dir,
                    candidate_cross_targets(source_dir, traces, same_app_only),
                    min_target_anom=args.min_target_anom,
                )
                for _, _, target_dir, shift in ranked[: args.topk_cross]:
                    out_dir = out_root / f"cross_entity_{shift_level}" / f"{source_dir.name}__to__{target_dir.name}"
                    ds_args = build_args(
                        source_dir=source_dir,
                        target_dir=target_dir,
                        out_dir=out_dir,
                        split_mode="search",
                        shift_level=shift_level,
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
                        seed=args.seed,
                    )
                    try:
                        meta = create_dataset(ds_args)
                        meta["candidate_pair_shift_precheck"] = shift
                        meta["same_app_only"] = bool(same_app_only)
                        manifest.append(meta)
                    except Exception as exc:
                        print(f"[WARN] cross {shift_level} {source_dir.name}->{target_dir.name}: {exc}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Total experiment folders: {len(manifest)}")


if __name__ == "__main__":
    main()
