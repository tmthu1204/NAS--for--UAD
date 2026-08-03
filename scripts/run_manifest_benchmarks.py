import argparse
import json
import math
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.run_domain_shift_case_matrix import (
    ensure_dir,
    float_or_none,
    read_json,
    run_mode_for_case_seed,
    sanitize_json,
    write_json,
)


KEY_METRICS = [
    "ap",
    "auroc",
    "f1_best",
    "f1_pot",
    "event_f1",
    "delay_mean",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run uad_source and adaptnas_combined directly from a built manifest.json."
    )
    ap.add_argument("--manifest", required=True, help="Path to manifest.json produced by a dataset builder.")
    ap.add_argument("--dataset_name", required=True, help="Display/output dataset name, e.g. smd or hai.")
    ap.add_argument(
        "--output_root",
        default=str(PROJ_ROOT / "outputs" / "benchmarks" / "cross_entity_hard_topPAD_global_top3_compare"),
        help="Root output directory for benchmark artifacts.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs_pretrain", type=int, default=10)
    ap.add_argument("--search_candidates", type=int, default=5)
    ap.add_argument("--nas_search_iters", type=int, default=3)
    ap.add_argument("--nas_compact_space", action="store_true")
    ap.add_argument("--nas_search_strategy", default="random")
    ap.add_argument("--nas_evo_parent_pool", type=int, default=3)
    ap.add_argument("--nas_evo_anchor_ratio", type=float, default=0.2)
    ap.add_argument("--nas_evo_mutation_steps", type=int, default=3)
    ap.add_argument("--nas_evo_cross_family_ratio", type=float, default=0.5)
    ap.add_argument("--nas_evo_random_ratio", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--combined_upper_gap", type=float, default=1.0)
    ap.add_argument("--combined_nas_topk_rerank", type=int, default=5)
    ap.add_argument("--combined_nas_proxy_rerank_topk", type=int, default=0)
    ap.add_argument("--combined_nas_fusion_rerank_topk", type=int, default=0)
    ap.add_argument("--combined_nas_fusion_upper_weight", type=float, default=1.0)
    ap.add_argument("--combined_nas_fusion_proxy_weight", type=float, default=1.0)
    ap.add_argument("--combined_nas_diverse_per_family", type=int, default=0)
    ap.add_argument("--combined_weight_tau", type=float, default=1.0)
    ap.add_argument("--combined_weight_w_min", type=float, default=0.05)
    ap.add_argument("--combined_weight_top_keep_ratio", type=float, default=1.0)
    ap.add_argument("--combined_search_candidate_warmup_steps", type=int, default=None)
    ap.add_argument("--combined_search_steps", type=int, default=None)
    ap.add_argument("--combined_final_candidate_warmup_steps", type=int, default=None)
    ap.add_argument("--combined_final_steps", type=int, default=None)
    ap.add_argument("--combined_final_patience", type=int, default=None)
    ap.add_argument("--oneclass_method", default="deepsvdd")
    ap.add_argument("--weighting_oneclass_method", default=None)
    ap.add_argument("--final_oneclass_method", default=None)
    ap.add_argument("--oneclass_epochs", type=int, default=10)
    ap.add_argument("--oneclass_final_epochs", type=int, default=20)
    ap.add_argument("--oneclass_lr", type=float, default=1e-3)
    ap.add_argument("--oneclass_batch_size", type=int, default=1024)
    ap.add_argument("--oneclass_max_fit", type=int, default=5000)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--iforest_n_estimators", type=int, default=100)
    ap.add_argument("--iforest_max_samples", default="auto")
    ap.add_argument("--iforest_contamination", default="auto")
    ap.add_argument("--iforest_max_features", type=float, default=1.0)
    ap.add_argument("--lof_hidden_dim", type=int, default=128)
    ap.add_argument("--lof_rep_dim", type=int, default=64)
    ap.add_argument("--lof_n_neighbors", type=int, default=20)
    ap.add_argument("--lof_metric", default="minkowski")
    ap.add_argument("--lof_p", type=int, default=2)
    ap.add_argument("--flow_hidden_dim", type=int, default=128)
    ap.add_argument("--flow_rep_dim", type=int, default=64)
    ap.add_argument("--flow_layers", type=int, default=4)
    ap.add_argument("--flow_warmup_epochs", type=int, default=2)
    ap.add_argument("--flow_scale_clip", type=float, default=2.0)
    ap.add_argument("--dagmm_hidden_dim", type=int, default=128)
    ap.add_argument("--dagmm_latent_dim", type=int, default=16)
    ap.add_argument("--dagmm_est_hidden_dim", type=int, default=64)
    ap.add_argument("--dagmm_components", type=int, default=3)
    ap.add_argument("--dagmm_lambda_energy", type=float, default=0.1)
    ap.add_argument("--dagmm_lambda_cov_diag", type=float, default=5e-3)
    ap.add_argument("--dagmm_warmup_epochs", type=int, default=2)
    ap.add_argument("--drocc_hidden_dim", type=int, default=128)
    ap.add_argument("--drocc_rep_dim", type=int, default=64)
    ap.add_argument("--drocc_radius", type=float, default=1.0)
    ap.add_argument("--drocc_gamma", type=float, default=2.0)
    ap.add_argument("--drocc_adv_steps", type=int, default=5)
    ap.add_argument("--drocc_adv_step_size", type=float, default=0.1)
    ap.add_argument("--drocc_warmup_epochs", type=int, default=2)
    ap.add_argument("--drocc_adv_weight", type=float, default=1.0)
    ap.add_argument("--drocc_compactness_weight", type=float, default=0.1)
    ap.add_argument("--ocsvm_nu", type=float, default=0.05)
    ap.add_argument("--ocsvm_kernel", default="rbf")
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
    ap.add_argument("--gmm_covariance_type", default="diag")
    ap.add_argument("--gmm_reg_covar", type=float, default=1e-4)
    ap.add_argument("--gmm_warmup_epochs", type=int, default=2)
    ap.add_argument("--proto_hidden_dim", type=int, default=128)
    ap.add_argument("--proto_rep_dim", type=int, default=64)
    ap.add_argument("--proto_count", type=int, default=4)
    ap.add_argument("--proto_separation_weight", type=float, default=0.1)
    ap.add_argument("--proto_separation_margin", type=float, default=1.0)
    ap.add_argument("--force_run", action="store_true")
    return ap.parse_args()


def mean_or_none(values):
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_oneclass_cli_args(args):
    cli = [
        "--oneclass_method", args.oneclass_method,
        "--oneclass_epochs", str(args.oneclass_epochs),
        "--oneclass_final_epochs", str(args.oneclass_final_epochs),
        "--oneclass_lr", str(args.oneclass_lr),
        "--oneclass_batch_size", str(args.oneclass_batch_size),
        "--oneclass_max_fit", str(args.oneclass_max_fit),
        "--knn_k", str(args.knn_k),
        "--iforest_n_estimators", str(args.iforest_n_estimators),
        "--iforest_max_samples", str(args.iforest_max_samples),
        "--iforest_contamination", str(args.iforest_contamination),
        "--iforest_max_features", str(args.iforest_max_features),
        "--lof_hidden_dim", str(args.lof_hidden_dim),
        "--lof_rep_dim", str(args.lof_rep_dim),
        "--lof_n_neighbors", str(args.lof_n_neighbors),
        "--lof_metric", str(args.lof_metric),
        "--lof_p", str(args.lof_p),
        "--flow_hidden_dim", str(args.flow_hidden_dim),
        "--flow_rep_dim", str(args.flow_rep_dim),
        "--flow_layers", str(args.flow_layers),
        "--flow_warmup_epochs", str(args.flow_warmup_epochs),
        "--flow_scale_clip", str(args.flow_scale_clip),
        "--dagmm_hidden_dim", str(args.dagmm_hidden_dim),
        "--dagmm_latent_dim", str(args.dagmm_latent_dim),
        "--dagmm_est_hidden_dim", str(args.dagmm_est_hidden_dim),
        "--dagmm_components", str(args.dagmm_components),
        "--dagmm_lambda_energy", str(args.dagmm_lambda_energy),
        "--dagmm_lambda_cov_diag", str(args.dagmm_lambda_cov_diag),
        "--dagmm_warmup_epochs", str(args.dagmm_warmup_epochs),
        "--drocc_hidden_dim", str(args.drocc_hidden_dim),
        "--drocc_rep_dim", str(args.drocc_rep_dim),
        "--drocc_radius", str(args.drocc_radius),
        "--drocc_gamma", str(args.drocc_gamma),
        "--drocc_adv_steps", str(args.drocc_adv_steps),
        "--drocc_adv_step_size", str(args.drocc_adv_step_size),
        "--drocc_warmup_epochs", str(args.drocc_warmup_epochs),
        "--drocc_adv_weight", str(args.drocc_adv_weight),
        "--drocc_compactness_weight", str(args.drocc_compactness_weight),
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
    if args.weighting_oneclass_method:
        cli.extend(["--weighting_oneclass_method", args.weighting_oneclass_method])
    if args.final_oneclass_method:
        cli.extend(["--final_oneclass_method", args.final_oneclass_method])
    return cli


def build_pipeline_extra_args(args):
    cli = []
    for name in (
        "combined_search_candidate_warmup_steps",
        "combined_search_steps",
        "combined_final_candidate_warmup_steps",
        "combined_final_steps",
        "combined_final_patience",
    ):
        value = getattr(args, name)
        if value is not None:
            cli.extend([f"--{name}", str(value)])
    return cli


def extract_combined_baseline_info(summary: dict):
    selection_strategy = str(summary.get("selection_strategy") or "best_by_val_auroc")
    all_entries = summary.get("all") or []
    by_arch = {}
    by_arch_selection = {}
    for entry in all_entries:
        arch_name = entry.get("arch_name")
        if not arch_name:
            continue
        by_arch[arch_name] = entry.get("metrics_uad") or {}
        by_arch_selection[arch_name] = entry.get("selection_metrics_uad") or {}

    best_fixed_name = None
    best_fixed_metrics = None
    best_fixed_selection_metrics = None
    best_fixed_val_auroc = float("-inf")
    for arch_name, metrics in by_arch.items():
        if not arch_name.startswith("Base_"):
            continue
        selection_metrics = by_arch_selection.get(arch_name) or {}
        auroc = float_or_none(selection_metrics.get("auroc"))
        if auroc is None:
            auroc = float_or_none(metrics.get("auroc"))
        if auroc is not None and auroc > best_fixed_val_auroc:
            best_fixed_val_auroc = auroc
            best_fixed_name = arch_name
            best_fixed_metrics = metrics
            best_fixed_selection_metrics = selection_metrics

    best_by_val = summary.get("best_by_val_auroc") or summary.get("best_by_auroc") or {}
    selected_entry = best_by_val
    best_nas_by_val = summary.get("best_nas_by_val_auroc") or {}
    if not best_nas_by_val:
        best_nas_name = None
        best_nas_metrics = None
        best_nas_selection_metrics = None
        best_nas_val_auroc = float("-inf")
        for arch_name, metrics in by_arch.items():
            if not arch_name.startswith("NAS_"):
                continue
            selection_metrics = by_arch_selection.get(arch_name) or {}
            auroc = float_or_none(selection_metrics.get("auroc"))
            if auroc is None:
                auroc = float_or_none(metrics.get("auroc"))
            if auroc is not None and auroc > best_nas_val_auroc:
                best_nas_val_auroc = auroc
                best_nas_name = arch_name
                best_nas_metrics = metrics
                best_nas_selection_metrics = selection_metrics
        best_nas_by_val = {
            "arch_name": best_nas_name,
            "metrics_uad": best_nas_metrics or {},
            "selection_metrics_uad": best_nas_selection_metrics or {},
        }
    return {
        "selection_strategy": selection_strategy,
        "winner_arch": selected_entry.get("arch_name"),
        "winner_metrics": selected_entry.get("metrics_uad") or {},
        "winner_selection_metrics": selected_entry.get("selection_metrics_uad") or {},
        "best_nas_arch": best_nas_by_val.get("arch_name"),
        "best_nas_metrics": best_nas_by_val.get("metrics_uad") or {},
        "best_nas_selection_metrics": best_nas_by_val.get("selection_metrics_uad") or {},
        "nas_metrics": best_nas_by_val.get("metrics_uad") or {},
        "nas_selection_metrics": best_nas_by_val.get("selection_metrics_uad") or {},
        "best_fixed_arch": best_fixed_name,
        "best_fixed_metrics": best_fixed_metrics or {},
        "best_fixed_selection_metrics": best_fixed_selection_metrics or {},
        "all_by_arch": by_arch,
        "all_selection_by_arch": by_arch_selection,
    }


def build_case_row(meta, source_res, combined_res, combined_summary):
    meta = dict(meta)
    source_metrics = source_res.get("metrics_uad") or {}
    combined_metrics = combined_res.get("metrics_uad") or {}
    baseline_info = extract_combined_baseline_info(combined_summary)
    best_fixed_metrics = baseline_info["best_fixed_metrics"]
    best_nas_metrics = baseline_info["best_nas_metrics"]
    pair_id = f"{meta['source_entity']}__to__{meta['target_entity']}"
    row = {
        "pair_id": pair_id,
        "source_entity": meta["source_entity"],
        "target_entity": meta["target_entity"],
        "shift_level": meta.get("selected_shift_level") or meta.get("shift_level"),
        "global_pad_rank": meta.get("global_pad_rank"),
        "pad_value": (meta.get("candidate_pair_shift_precheck") or {}).get("pad_value"),
        "rule_selection_score": (
            (meta.get("less_fixed_friendly") or {}).get("refined_score")
            or (meta.get("learnable_shift_val_rich") or {}).get("score")
        ),
        "selection_strategy": baseline_info["selection_strategy"],
        "winner_arch": baseline_info["winner_arch"],
        "best_fixed_arch": baseline_info["best_fixed_arch"],
        "best_nas_arch": baseline_info["best_nas_arch"],
        "report_split": combined_summary.get("report_split"),
        "selection_split": combined_summary.get("selection_split"),
        "train_normal_count": meta.get("train_normal_count"),
        "target_pool_count": meta.get("target_pool_count"),
        "val_count": meta.get("val_count"),
        "test_count": meta.get("test_count"),
        "target_pool_hidden_anomaly_count": meta.get("target_pool_hidden_anomaly_count"),
        "val_anomaly_count": meta.get("val_anomaly_count"),
        "test_anomaly_count": meta.get("test_anomaly_count"),
        "uad_source": source_metrics,
        "combined": combined_metrics,
        "best_nas": best_nas_metrics,
        "best_fixed": best_fixed_metrics,
        "nas_bestarch": best_nas_metrics,
        "nas_surrogate_alignment": combined_summary.get("nas_surrogate_alignment") or {},
        "delta_combined_minus_source": {},
        "delta_nas_minus_best_fixed": {},
    }
    for metric in KEY_METRICS:
        src_val = float_or_none(source_metrics.get(metric))
        cmb_val = float_or_none(combined_metrics.get(metric))
        fix_val = float_or_none(best_fixed_metrics.get(metric))
        nas_val = float_or_none(best_nas_metrics.get(metric))
        row["delta_combined_minus_source"][metric] = None if src_val is None or cmb_val is None else float(cmb_val - src_val)
        row["delta_nas_minus_best_fixed"][metric] = None if fix_val is None or nas_val is None else float(nas_val - fix_val)
    return row


def write_report(path: Path, dataset_name: str, rows):
    lines = [f"# {dataset_name} manifest benchmark", ""]
    lines.append("| Pair | PAD rank | PAD | Winner | Best fixed | Best NAS | uad_source (AP/AUROC/F1) | combined (AP/AUROC/F1) | best_NAS (AP/AUROC/F1) | best_fixed (AP/AUROC/F1) | d_combined-source AUROC | d_bestNAS-fixed AUROC |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        us = row["uad_source"]
        cmb = row["combined"]
        nas = row["best_nas"]
        fix = row["best_fixed"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["pair_id"],
                    str(row.get("global_pad_rank", "")),
                    fmt(row.get("pad_value")),
                    str(row.get("winner_arch", "")),
                    str(row.get("best_fixed_arch", "")),
                    str(row.get("best_nas_arch", "")),
                    f"{fmt(us.get('ap'))} / {fmt(us.get('auroc'))} / {fmt(us.get('f1_best'))}",
                    f"{fmt(cmb.get('ap'))} / {fmt(cmb.get('auroc'))} / {fmt(cmb.get('f1_best'))}",
                    f"{fmt(nas.get('ap'))} / {fmt(nas.get('auroc'))} / {fmt(nas.get('f1_best'))}",
                    f"{fmt(fix.get('ap'))} / {fmt(fix.get('auroc'))} / {fmt(fix.get('f1_best'))}",
                    fmt(row["delta_combined_minus_source"].get("auroc")),
                    fmt(row["delta_nas_minus_best_fixed"].get("auroc")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Means")
    lines.append("")
    lines.append("| Metric | uad_source | combined | best_NAS | best_fixed |")
    lines.append("| --- | --- | --- | --- | --- |")
    for metric in KEY_METRICS:
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    fmt(mean_or_none([float_or_none(row["uad_source"].get(metric)) for row in rows])),
                    fmt(mean_or_none([float_or_none(row["combined"].get(metric)) for row in rows])),
                    fmt(mean_or_none([float_or_none(row["best_nas"].get(metric)) for row in rows])),
                    fmt(mean_or_none([float_or_none(row["best_fixed"].get(metric)) for row in rows])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## NAS Surrogate Alignment")
    lines.append("")
    lines.append("| Pair | Best NAS | NAS eval count | corr(-upper_obj, val AUROC) | corr(-proxy_obj, val AUROC) | upper top1 hit | proxy top1 hit |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        align = row.get("nas_surrogate_alignment") or {}
        upper = align.get("upper_obj") or {}
        proxy = align.get("score_proxy_obj") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    row["pair_id"],
                    str(row.get("best_nas_arch", "")),
                    str((upper.get("n") if upper.get("n") is not None else proxy.get("n")) or ""),
                    fmt(upper.get("spearman_neg_obj_vs_val_auroc")),
                    fmt(proxy.get("spearman_neg_obj_vs_val_auroc")),
                    "Y" if upper.get("top1_matches_best_val") else "N" if upper.get("top1_matches_best_val") is not None else "",
                    "Y" if proxy.get("top1_matches_best_val") else "N" if proxy.get("top1_matches_best_val") is not None else "",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return ""


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Manifest is empty or invalid.")

    py = sys.executable
    output_root = Path(args.output_root) / args.dataset_name
    ensure_dir(output_root)
    oneclass_cli_args = build_oneclass_cli_args(args)
    pipeline_extra_args = build_pipeline_extra_args(args)

    rows = []
    for meta in manifest:
        split_dir = Path(meta["out_dir"])
        pair_id = f"{meta['source_entity']}__to__{meta['target_entity']}"
        run_dir = output_root / pair_id
        run_mode_for_case_seed(
            py,
            split_dir,
            run_dir,
            mode="uad_source",
            seed=args.seed,
            epochs_pretrain=args.epochs_pretrain,
            search_candidates=args.search_candidates,
            nas_search_iters=args.nas_search_iters,
            nas_compact_space=args.nas_compact_space,
            nas_search_strategy=args.nas_search_strategy,
            nas_evo_parent_pool=args.nas_evo_parent_pool,
            nas_evo_anchor_ratio=args.nas_evo_anchor_ratio,
            nas_evo_mutation_steps=args.nas_evo_mutation_steps,
            nas_evo_cross_family_ratio=args.nas_evo_cross_family_ratio,
            nas_evo_random_ratio=args.nas_evo_random_ratio,
            batch_size=args.batch_size,
            device=args.device,
            combined_upper_gap=args.combined_upper_gap,
            combined_nas_topk_rerank=args.combined_nas_topk_rerank,
            combined_nas_proxy_rerank_topk=args.combined_nas_proxy_rerank_topk,
            combined_nas_fusion_rerank_topk=args.combined_nas_fusion_rerank_topk,
            combined_nas_fusion_upper_weight=args.combined_nas_fusion_upper_weight,
            combined_nas_fusion_proxy_weight=args.combined_nas_fusion_proxy_weight,
            combined_nas_diverse_per_family=args.combined_nas_diverse_per_family,
            combined_weight_tau=args.combined_weight_tau,
            combined_weight_w_min=args.combined_weight_w_min,
            combined_weight_top_keep_ratio=args.combined_weight_top_keep_ratio,
            oneclass_cli_args=oneclass_cli_args,
            force=args.force_run,
            pipeline_extra_args=pipeline_extra_args,
        )
        run_mode_for_case_seed(
            py,
            split_dir,
            run_dir,
            mode="adaptnas_combined",
            seed=args.seed,
            epochs_pretrain=args.epochs_pretrain,
            search_candidates=args.search_candidates,
            nas_search_iters=args.nas_search_iters,
            nas_compact_space=args.nas_compact_space,
            nas_search_strategy=args.nas_search_strategy,
            nas_evo_parent_pool=args.nas_evo_parent_pool,
            nas_evo_anchor_ratio=args.nas_evo_anchor_ratio,
            nas_evo_mutation_steps=args.nas_evo_mutation_steps,
            nas_evo_cross_family_ratio=args.nas_evo_cross_family_ratio,
            nas_evo_random_ratio=args.nas_evo_random_ratio,
            batch_size=args.batch_size,
            device=args.device,
            combined_upper_gap=args.combined_upper_gap,
            combined_nas_topk_rerank=args.combined_nas_topk_rerank,
            combined_nas_proxy_rerank_topk=args.combined_nas_proxy_rerank_topk,
            combined_nas_fusion_rerank_topk=args.combined_nas_fusion_rerank_topk,
            combined_nas_fusion_upper_weight=args.combined_nas_fusion_upper_weight,
            combined_nas_fusion_proxy_weight=args.combined_nas_fusion_proxy_weight,
            combined_nas_diverse_per_family=args.combined_nas_diverse_per_family,
            combined_weight_tau=args.combined_weight_tau,
            combined_weight_w_min=args.combined_weight_w_min,
            combined_weight_top_keep_ratio=args.combined_weight_top_keep_ratio,
            oneclass_cli_args=oneclass_cli_args,
            force=args.force_run,
            pipeline_extra_args=pipeline_extra_args,
        )
        rows.append(
            build_case_row(
                meta,
                read_json(run_dir / "uad_source_results.json"),
                read_json(run_dir / "adaptnas_combined_results.json"),
                read_json(run_dir / "adaptnas_combined_baselines_summary.json"),
            )
        )

    summary = {
        "dataset_name": args.dataset_name,
        "manifest": str(manifest_path),
        "seed": args.seed,
        "device": args.device,
        "settings": {
            "epochs_pretrain": args.epochs_pretrain,
            "search_candidates": args.search_candidates,
            "nas_search_iters": args.nas_search_iters,
            "nas_compact_space": args.nas_compact_space,
            "nas_search_strategy": args.nas_search_strategy,
            "nas_evo_parent_pool": args.nas_evo_parent_pool,
            "nas_evo_anchor_ratio": args.nas_evo_anchor_ratio,
            "nas_evo_mutation_steps": args.nas_evo_mutation_steps,
            "nas_evo_cross_family_ratio": args.nas_evo_cross_family_ratio,
            "nas_evo_random_ratio": args.nas_evo_random_ratio,
            "batch_size": args.batch_size,
            "combined_upper_gap": args.combined_upper_gap,
            "combined_nas_topk_rerank": args.combined_nas_topk_rerank,
            "combined_nas_proxy_rerank_topk": args.combined_nas_proxy_rerank_topk,
            "combined_nas_fusion_rerank_topk": args.combined_nas_fusion_rerank_topk,
            "combined_nas_fusion_upper_weight": args.combined_nas_fusion_upper_weight,
            "combined_nas_fusion_proxy_weight": args.combined_nas_fusion_proxy_weight,
            "combined_nas_diverse_per_family": args.combined_nas_diverse_per_family,
            "combined_weight_tau": args.combined_weight_tau,
            "combined_weight_w_min": args.combined_weight_w_min,
            "combined_weight_top_keep_ratio": args.combined_weight_top_keep_ratio,
            "combined_search_candidate_warmup_steps": args.combined_search_candidate_warmup_steps,
            "combined_search_steps": args.combined_search_steps,
            "combined_final_candidate_warmup_steps": args.combined_final_candidate_warmup_steps,
            "combined_final_steps": args.combined_final_steps,
            "combined_final_patience": args.combined_final_patience,
            "oneclass_method": args.oneclass_method,
            "weighting_oneclass_method": args.weighting_oneclass_method,
            "final_oneclass_method": args.final_oneclass_method,
            "oneclass_epochs": args.oneclass_epochs,
            "oneclass_final_epochs": args.oneclass_final_epochs,
            "oneclass_batch_size": args.oneclass_batch_size,
            "oneclass_max_fit": args.oneclass_max_fit,
            "iforest_n_estimators": args.iforest_n_estimators,
            "iforest_max_samples": args.iforest_max_samples,
            "iforest_contamination": args.iforest_contamination,
            "iforest_max_features": args.iforest_max_features,
            "lof_hidden_dim": args.lof_hidden_dim,
            "lof_rep_dim": args.lof_rep_dim,
            "lof_n_neighbors": args.lof_n_neighbors,
            "lof_metric": args.lof_metric,
            "lof_p": args.lof_p,
            "flow_hidden_dim": args.flow_hidden_dim,
            "flow_rep_dim": args.flow_rep_dim,
            "flow_layers": args.flow_layers,
            "flow_warmup_epochs": args.flow_warmup_epochs,
            "flow_scale_clip": args.flow_scale_clip,
            "dagmm_hidden_dim": args.dagmm_hidden_dim,
            "dagmm_latent_dim": args.dagmm_latent_dim,
            "dagmm_est_hidden_dim": args.dagmm_est_hidden_dim,
            "dagmm_components": args.dagmm_components,
            "dagmm_lambda_energy": args.dagmm_lambda_energy,
            "dagmm_lambda_cov_diag": args.dagmm_lambda_cov_diag,
            "dagmm_warmup_epochs": args.dagmm_warmup_epochs,
            "drocc_hidden_dim": args.drocc_hidden_dim,
            "drocc_rep_dim": args.drocc_rep_dim,
            "drocc_radius": args.drocc_radius,
            "drocc_gamma": args.drocc_gamma,
            "drocc_adv_steps": args.drocc_adv_steps,
            "drocc_adv_step_size": args.drocc_adv_step_size,
            "drocc_warmup_epochs": args.drocc_warmup_epochs,
            "drocc_adv_weight": args.drocc_adv_weight,
            "drocc_compactness_weight": args.drocc_compactness_weight,
        },
        "rows": rows,
    }
    write_json(output_root / "summary.json", summary)
    write_report(output_root / "REPORT.md", args.dataset_name, rows)
    print(f"[DONE] Saved benchmark summary to {output_root}")


if __name__ == "__main__":
    main()
