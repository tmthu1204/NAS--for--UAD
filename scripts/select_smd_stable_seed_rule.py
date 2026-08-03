import argparse
import json
import math
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.build_pair_rule_common import public_rows, safe_float, save_json, write_table_csv, write_table_markdown


RULE_NAME = "cross_entity_stable_seed_region"


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def format_float(value, digits=4):
    value = safe_float(value, default=float("nan"))
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def metric_delta(row: dict, left_key: str, right_key: str, metric: str) -> float:
    left = safe_float(((row.get(left_key) or {}).get(metric)), default=float("nan"))
    right = safe_float(((row.get(right_key) or {}).get(metric)), default=float("nan"))
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return float(left - right)


def merge_rows(manifest_rows, build_summary_rows, benchmark_rows):
    manifest_by_pair = {f"{row['source_entity']}__to__{row['target_entity']}": row for row in manifest_rows}
    build_by_pair = {
        row["pair_id"]: row
        for row in build_summary_rows
        if row.get("build_success")
    }
    benchmark_by_pair = {row["pair_id"]: row for row in benchmark_rows}

    merged = []
    for pair_id, manifest_row in manifest_by_pair.items():
        build_row = build_by_pair.get(pair_id)
        benchmark_row = benchmark_by_pair.get(pair_id)
        if build_row is None or benchmark_row is None:
            continue

        val_count = int(manifest_row.get("val_count", 0) or 0)
        val_anom = int(manifest_row.get("val_anomaly_count", 0) or 0)
        test_count = int(manifest_row.get("test_count", 0) or 0)
        test_anom = int(manifest_row.get("test_anomaly_count", 0) or 0)
        delta_combined = metric_delta(benchmark_row, "combined", "uad_source", "auroc")
        delta_nas = metric_delta(benchmark_row, "best_nas", "best_fixed", "auroc")

        merged.append(
            {
                "pair_id": pair_id,
                "source_entity": manifest_row["source_entity"],
                "target_entity": manifest_row["target_entity"],
                "shift_level": manifest_row.get("selected_shift_level") or manifest_row.get("shift_level"),
                "global_pad_rank": int(manifest_row.get("global_pad_rank") or build_row.get("global_pad_rank") or 10**9),
                "pad_value": safe_float(build_row.get("pad_value")),
                "precheck_feature_mean_l2": safe_float(build_row.get("pad_feature_mean_l2")),
                "target_pool_count": int(manifest_row.get("target_pool_count", 0) or 0),
                "target_pool_hidden_anomaly_count": int(manifest_row.get("target_pool_hidden_anomaly_count", 0) or 0),
                "target_pool_hidden_anomaly_ratio": safe_float(manifest_row.get("target_pool_hidden_anomaly_ratio")),
                "val_count": val_count,
                "val_anomaly_count": val_anom,
                "val_anomaly_ratio": (float(val_anom / val_count) if val_count > 0 else float("nan")),
                "test_count": test_count,
                "test_anomaly_count": test_anom,
                "test_anomaly_ratio": (float(test_anom / test_count) if test_count > 0 else float("nan")),
                "delta_combined_minus_source_auroc": delta_combined,
                "delta_nas_minus_fixed_auroc": delta_nas,
                "is_good": bool(
                    math.isfinite(delta_combined)
                    and math.isfinite(delta_nas)
                    and delta_combined > 0.0
                    and delta_nas > 0.0
                ),
                "manifest_row": manifest_row,
                "benchmark_row": benchmark_row,
            }
        )

    merged.sort(key=lambda row: (row["global_pad_rank"], row["pair_id"]))
    return merged


def parse_shift_levels(raw: str):
    return {item.strip() for item in raw.split(",") if item.strip()}


def matches_rule(row: dict, args) -> bool:
    if row.get("shift_level") not in parse_shift_levels(args.shift_levels):
        return False
    if row["global_pad_rank"] < args.min_global_pad_rank or row["global_pad_rank"] > args.max_global_pad_rank:
        return False

    precheck = safe_float(row.get("precheck_feature_mean_l2"), default=float("nan"))
    if not math.isfinite(precheck) or precheck < args.min_precheck_l2 or precheck > args.max_precheck_l2:
        return False

    pool_ratio = safe_float(row.get("target_pool_hidden_anomaly_ratio"), default=float("nan"))
    if not math.isfinite(pool_ratio) or pool_ratio > args.max_pool_anom_ratio:
        return False

    val_count = int(row.get("val_count", 0))
    if val_count < args.min_val_count or val_count > args.max_val_count:
        return False

    val_anom = int(row.get("val_anomaly_count", 0))
    if val_anom < args.min_val_anomaly_count or val_anom > args.max_val_anomaly_count:
        return False

    val_ratio = safe_float(row.get("val_anomaly_ratio"), default=float("nan"))
    if not math.isfinite(val_ratio) or val_ratio < args.min_val_anomaly_ratio or val_ratio > args.max_val_anomaly_ratio:
        return False

    return True


def evaluate_selected(selected_rows, all_rows):
    tp = sum(1 for row in selected_rows if row["is_good"])
    fp = len(selected_rows) - tp
    total_good = sum(1 for row in all_rows if row["is_good"])
    fn = total_good - tp
    precision = float(tp / len(selected_rows)) if selected_rows else 0.0
    recall = float(tp / total_good) if total_good else 0.0
    return {
        "selected_count": int(len(selected_rows)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "total_good": int(total_good),
        "precision": float(precision),
        "recall_over_good": float(recall),
        "mean_delta_combined_minus_source_auroc": mean_value(
            row["delta_combined_minus_source_auroc"] for row in selected_rows
        ),
        "mean_delta_nas_minus_fixed_auroc": mean_value(
            row["delta_nas_minus_fixed_auroc"] for row in selected_rows
        ),
    }


def mean_value(values):
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_rule_text(args):
    return (
        f"shift_level in [{args.shift_levels}], "
        f"{args.min_global_pad_rank} <= global_pad_rank <= {args.max_global_pad_rank}, "
        f"{args.min_precheck_l2:.4f} <= precheck_feature_mean_l2 <= {args.max_precheck_l2:.4f}, "
        f"target_pool_hidden_anomaly_ratio <= {args.max_pool_anom_ratio:.4f}, "
        f"{args.min_val_count} <= val_count <= {args.max_val_count}, "
        f"{args.min_val_anomaly_count} <= val_anomaly_count <= {args.max_val_anomaly_count}, "
        f"{args.min_val_anomaly_ratio:.4f} <= val_anomaly_ratio <= {args.max_val_anomaly_ratio:.4f}"
    )


def write_report(path: Path, summary: dict, selected_rows, all_rows):
    lines = [
        "# SMD Stable Seed Region Rule",
        "",
        "This experimental rule is derived from the stable seed region found from SMD top20 pairs that won in at least two backend configs.",
        "",
        "## Rule",
        "",
        f"`{summary['rule_text']}`",
        "",
        f"- Buildable + benchmarked pairs considered: {summary['n_rows_total']}",
        f"- Good pairs in benchmark universe: {summary['selected_stats']['total_good']}",
        f"- Selected pairs: {summary['selected_stats']['selected_count']}",
        f"- True positives: {summary['selected_stats']['tp']}",
        f"- False positives: {summary['selected_stats']['fp']}",
        f"- Precision: {format_float(summary['selected_stats']['precision'])}",
        f"- Recall over good pairs: {format_float(summary['selected_stats']['recall_over_good'])}",
        f"- Mean d(combined-source) AUROC on selected: {format_float(summary['selected_stats']['mean_delta_combined_minus_source_auroc'])}",
        f"- Mean d(NAS-fixed) AUROC on selected: {format_float(summary['selected_stats']['mean_delta_nas_minus_fixed_auroc'])}",
        "",
        "## Selected Pairs",
        "",
        "| PAD rank | Pair | Shift | precheck L2 | pool ratio | val count | val anom | val ratio | d combined-source AUROC | d NAS-fixed AUROC | good |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in selected_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["global_pad_rank"]),
                    row["pair_id"],
                    row["shift_level"],
                    format_float(row["precheck_feature_mean_l2"]),
                    format_float(row["target_pool_hidden_anomaly_ratio"]),
                    str(row["val_count"]),
                    str(row["val_anomaly_count"]),
                    format_float(row["val_anomaly_ratio"]),
                    format_float(row["delta_combined_minus_source_auroc"]),
                    format_float(row["delta_nas_minus_fixed_auroc"]),
                    "yes" if row["is_good"] else "no",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## All Pairs",
            "",
            "| PAD rank | Pair | Shift | precheck L2 | pool ratio | val count | val anom | val ratio | d combined-source AUROC | d NAS-fixed AUROC | good | selected |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    selected_pair_ids = {row["pair_id"] for row in selected_rows}
    for row in all_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["global_pad_rank"]),
                    row["pair_id"],
                    row["shift_level"],
                    format_float(row["precheck_feature_mean_l2"]),
                    format_float(row["target_pool_hidden_anomaly_ratio"]),
                    str(row["val_count"]),
                    str(row["val_anomaly_count"]),
                    format_float(row["val_anomaly_ratio"]),
                    format_float(row["delta_combined_minus_source_auroc"]),
                    format_float(row["delta_nas_minus_fixed_auroc"]),
                    "yes" if row["is_good"] else "no",
                    "yes" if row["pair_id"] in selected_pair_ids else "no",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Apply an experimental stable-seed region rule to all buildable SMD pairs and audit precision/recall."
    )
    ap.add_argument(
        "--manifest",
        default=str(PROJ_ROOT / "data" / "smd_experiments" / "cross_entity_all_pad_prefhard" / "manifest.json"),
    )
    ap.add_argument(
        "--build_summary",
        default=str(PROJ_ROOT / "data" / "smd_experiments" / "cross_entity_all_pad_prefhard" / "build_summary.json"),
    )
    ap.add_argument(
        "--benchmark_summary",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "smd_all_pad_prefhard_weighting_split_compare"
            / "smd_all_pad_prefhard_weight_gmm_final_svdd"
            / "summary.json"
        ),
    )
    ap.add_argument(
        "--out_dir",
        default=str(PROJ_ROOT / "data" / "smd_experiments" / RULE_NAME),
    )
    ap.add_argument("--shift_levels", default="hard,medium")
    ap.add_argument("--min_global_pad_rank", type=int, default=10)
    ap.add_argument("--max_global_pad_rank", type=int, default=16)
    ap.add_argument("--min_precheck_l2", type=float, default=5.0)
    ap.add_argument("--max_precheck_l2", type=float, default=7.3)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=0.07)
    ap.add_argument("--min_val_count", type=int, default=47)
    ap.add_argument("--max_val_count", type=int, default=67)
    ap.add_argument("--min_val_anomaly_count", type=int, default=7)
    ap.add_argument("--max_val_anomaly_count", type=int, default=15)
    ap.add_argument("--min_val_anomaly_ratio", type=float, default=0.145)
    ap.add_argument("--max_val_anomaly_ratio", type=float, default=0.224)
    ap.add_argument(
        "--seed_region_source",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "smd_top20_pad_weighting_split_compare"
            / "STABLE_GOOD_PAIRS_SEED.json"
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    manifest_rows = read_json(Path(args.manifest))
    build_summary_rows = read_json(Path(args.build_summary))
    benchmark_rows = read_json(Path(args.benchmark_summary)).get("rows") or []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_rows = merge_rows(manifest_rows, build_summary_rows, benchmark_rows)
    if not merged_rows:
        raise RuntimeError("No overlapping buildable + benchmarked SMD rows found.")

    selected_rows = [row for row in merged_rows if matches_rule(row, args)]
    selected_rows.sort(key=lambda row: (row["global_pad_rank"], row["pair_id"]))
    selected_pair_ids = {row["pair_id"] for row in selected_rows}

    selected_manifest = []
    for order, row in enumerate(selected_rows, start=1):
        meta = dict(row["manifest_row"])
        meta["selection_order"] = order
        meta["stable_seed_rule"] = {
            "rule_text": build_rule_text(args),
            "seed_region_source": args.seed_region_source,
        }
        meta["ranking_rule"] = RULE_NAME
        selected_manifest.append(meta)

    selected_stats = evaluate_selected(selected_rows, merged_rows)
    summary = {
        "rule_name": RULE_NAME,
        "dataset": "smd",
        "source_manifest": str(Path(args.manifest)),
        "source_build_summary": str(Path(args.build_summary)),
        "source_benchmark_summary": str(Path(args.benchmark_summary)),
        "seed_region_source": args.seed_region_source,
        "good_definition": {
            "combined_metric": "auroc",
            "nas_metric": "auroc",
            "condition": "combined.auroc > uad_source.auroc and best_nas.auroc > best_fixed.auroc",
        },
        "rule": {
            "shift_levels": sorted(parse_shift_levels(args.shift_levels)),
            "min_global_pad_rank": args.min_global_pad_rank,
            "max_global_pad_rank": args.max_global_pad_rank,
            "min_precheck_l2": args.min_precheck_l2,
            "max_precheck_l2": args.max_precheck_l2,
            "max_pool_anom_ratio": args.max_pool_anom_ratio,
            "min_val_count": args.min_val_count,
            "max_val_count": args.max_val_count,
            "min_val_anomaly_count": args.min_val_anomaly_count,
            "max_val_anomaly_count": args.max_val_anomaly_count,
            "min_val_anomaly_ratio": args.min_val_anomaly_ratio,
            "max_val_anomaly_ratio": args.max_val_anomaly_ratio,
        },
        "rule_text": build_rule_text(args),
        "n_rows_total": int(len(merged_rows)),
        "selected_stats": selected_stats,
        "rows": public_rows(
            [
                {
                    **row,
                    "selected": row["pair_id"] in selected_pair_ids,
                    "manifest_row": None,
                    "benchmark_row": None,
                }
                for row in merged_rows
            ]
        ),
    }

    columns = [
        "global_pad_rank",
        "pair_id",
        "shift_level",
        "pad_value",
        "precheck_feature_mean_l2",
        "target_pool_hidden_anomaly_ratio",
        "val_count",
        "val_anomaly_count",
        "val_anomaly_ratio",
        "test_count",
        "test_anomaly_count",
        "test_anomaly_ratio",
        "delta_combined_minus_source_auroc",
        "delta_nas_minus_fixed_auroc",
        "is_good",
        "selected",
    ]
    table_rows = [
        {
            **row,
            "selected": row["pair_id"] in selected_pair_ids,
        }
        for row in merged_rows
    ]

    save_json(out_dir / "manifest.json", selected_manifest)
    save_json(out_dir / "selection_summary.json", summary)
    write_table_csv(out_dir / "selection_summary.csv", table_rows, columns)
    write_table_markdown(
        out_dir / "selection_summary.md",
        table_rows,
        columns,
        title="SMD Stable Seed Region Selection",
    )
    write_report(out_dir / "REPORT.md", summary, selected_rows, merged_rows)

    print(f"[DONE] Saved manifest: {out_dir / 'manifest.json'}")
    print(f"[DONE] Selected pairs: {selected_stats['selected_count']} / {len(merged_rows)}")
    print(
        "[DONE] Precision / Recall = "
        f"{selected_stats['precision']:.4f} / {selected_stats['recall_over_good']:.4f}"
    )
    print(f"[DONE] Rule: {summary['rule_text']}")


if __name__ == "__main__":
    main()
