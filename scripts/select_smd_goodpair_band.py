import argparse
import json
import math
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.build_pair_rule_common import (
    public_rows,
    safe_float,
    save_json,
    write_table_csv,
    write_table_markdown,
)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric_delta(row: dict, left_key: str, right_key: str, metric: str) -> float:
    left = safe_float(((row.get(left_key) or {}).get(metric)), default=float("nan"))
    right = safe_float(((row.get(right_key) or {}).get(metric)), default=float("nan"))
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return float(left - right)


def evaluate_selected(selected_rows):
    tp = sum(1 for row in selected_rows if row["is_good"])
    fp = len(selected_rows) - tp
    precision = float(tp / len(selected_rows)) if selected_rows else 0.0
    mean_delta_combined = mean_value(row["delta_combined_minus_source_auroc"] for row in selected_rows)
    mean_delta_nas = mean_value(row["delta_nas_minus_fixed_auroc"] for row in selected_rows)
    return {
        "selected_count": int(len(selected_rows)),
        "tp": int(tp),
        "fp": int(fp),
        "precision": float(precision),
        "mean_delta_combined_minus_source_auroc": mean_delta_combined,
        "mean_delta_nas_minus_fixed_auroc": mean_delta_nas,
    }


def mean_value(values):
    vals = [float(v) for v in values if is_finite_number(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def format_float(value, digits=4):
    if value is None:
        return ""
    value = safe_float(value, default=float("nan"))
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def merge_rows(manifest_rows, build_summary_rows, benchmark_rows):
    manifest_by_pair = {f"{row['source_entity']}__to__{row['target_entity']}": row for row in manifest_rows}
    benchmark_by_pair = {row["pair_id"]: row for row in benchmark_rows}
    rows = []
    for build_row in build_summary_rows:
        pair_id = build_row["pair_id"]
        if not build_row.get("build_success"):
            continue
        if pair_id not in manifest_by_pair or pair_id not in benchmark_by_pair:
            continue
        benchmark_row = benchmark_by_pair[pair_id]
        manifest_row = manifest_by_pair[pair_id]
        delta_combined = metric_delta(benchmark_row, "combined", "uad_source", "auroc")
        delta_nas = metric_delta(benchmark_row, "best_nas", "best_fixed", "auroc")
        rows.append(
            {
                "pair_id": pair_id,
                "source_entity": build_row["source_entity"],
                "target_entity": build_row["target_entity"],
                "shift_level": build_row["selected_shift_level"],
                "global_pad_rank": int(build_row["global_pad_rank"]),
                "pad_value": safe_float(build_row.get("pad_value")),
                "precheck_feature_mean_l2": safe_float(build_row.get("pad_feature_mean_l2")),
                "target_pool_hidden_anomaly_ratio": safe_float(
                    manifest_row.get("target_pool_hidden_anomaly_ratio")
                ),
                "val_count": int(manifest_row.get("val_count", 0)),
                "val_anomaly_count": int(manifest_row.get("val_anomaly_count", 0)),
                "val_anomaly_ratio": safe_float(
                    (
                        manifest_row.get("val_anomaly_count", 0) / manifest_row.get("val_count", 0)
                        if manifest_row.get("val_count", 0)
                        else float("nan")
                    )
                ),
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
    rows.sort(key=lambda row: row["global_pad_rank"])
    return rows


def matches_rule(row: dict, rule: dict) -> bool:
    if rule.get("precheck_feature_mean_l2_low") is not None:
        if safe_float(row.get("precheck_feature_mean_l2"), default=float("nan")) < rule["precheck_feature_mean_l2_low"]:
            return False
    if rule.get("precheck_feature_mean_l2_high") is not None:
        if safe_float(row.get("precheck_feature_mean_l2"), default=float("nan")) > rule["precheck_feature_mean_l2_high"]:
            return False
    if rule.get("target_pool_hidden_anomaly_ratio_max") is not None:
        if safe_float(row.get("target_pool_hidden_anomaly_ratio"), default=float("nan")) > rule["target_pool_hidden_anomaly_ratio_max"]:
            return False
    if rule.get("val_count_min") is not None:
        if int(row.get("val_count", 0)) < int(rule["val_count_min"]):
            return False
    if rule.get("val_anomaly_count_min") is not None:
        if int(row.get("val_anomaly_count", 0)) < int(rule["val_anomaly_count_min"]):
            return False
    if rule.get("val_anomaly_ratio_max") is not None:
        if safe_float(row.get("val_anomaly_ratio"), default=float("nan")) > rule["val_anomaly_ratio_max"]:
            return False
    return True


def select_rows(rows, rule):
    return [row for row in rows if matches_rule(row, rule)]


def search_best_rule(rows):
    feature_values = {
        "precheck_feature_mean_l2": sorted(
            {row["precheck_feature_mean_l2"] for row in rows if is_finite_number(row["precheck_feature_mean_l2"])}
        ),
        "target_pool_hidden_anomaly_ratio": sorted(
            {row["target_pool_hidden_anomaly_ratio"] for row in rows if is_finite_number(row["target_pool_hidden_anomaly_ratio"])}
        ),
        "val_count": sorted({int(row["val_count"]) for row in rows}),
        "val_anomaly_count": sorted({int(row["val_anomaly_count"]) for row in rows}),
        "val_anomaly_ratio": sorted(
            {row["val_anomaly_ratio"] for row in rows if is_finite_number(row["val_anomaly_ratio"])}
        ),
    }

    best = None
    for l2_low in feature_values["precheck_feature_mean_l2"]:
        for l2_high in feature_values["precheck_feature_mean_l2"]:
            if l2_high < l2_low:
                continue
            for pool_max in feature_values["target_pool_hidden_anomaly_ratio"]:
                for val_count_min in feature_values["val_count"]:
                    for val_anomaly_count_min in feature_values["val_anomaly_count"]:
                        for val_anomaly_ratio_max in feature_values["val_anomaly_ratio"]:
                            rule = {
                                "precheck_feature_mean_l2_low": float(l2_low),
                                "precheck_feature_mean_l2_high": float(l2_high),
                                "target_pool_hidden_anomaly_ratio_max": float(pool_max),
                                "val_count_min": int(val_count_min),
                                "val_anomaly_count_min": int(val_anomaly_count_min),
                                "val_anomaly_ratio_max": float(val_anomaly_ratio_max),
                            }
                            selected = select_rows(rows, rule)
                            if not selected:
                                continue
                            stats = evaluate_selected(selected)
                            rank_sum = sum(row["global_pad_rank"] for row in selected)
                            score = (
                                stats["precision"],
                                stats["tp"],
                                -stats["fp"],
                                stats["selected_count"],
                                -rank_sum,
                                stats["mean_delta_combined_minus_source_auroc"] or float("-inf"),
                                stats["mean_delta_nas_minus_fixed_auroc"] or float("-inf"),
                            )
                            candidate = {
                                "rule": rule,
                                "selected_rows": selected,
                                "stats": stats,
                                "score": score,
                            }
                            if best is None or candidate["score"] > best["score"]:
                                best = candidate
    if best is None:
        raise RuntimeError("No valid rule found.")
    return best


def simplify_rule(rows, selected_rows, initial_rule):
    target_pair_ids = {row["pair_id"] for row in selected_rows}
    simplified = dict(initial_rule)
    field_order = [
        "target_pool_hidden_anomaly_ratio_max",
        "val_count_min",
        "val_anomaly_count_min",
        "val_anomaly_ratio_max",
        "precheck_feature_mean_l2_low",
        "precheck_feature_mean_l2_high",
    ]
    for field in field_order:
        candidate = dict(simplified)
        candidate[field] = None
        candidate_pair_ids = {row["pair_id"] for row in select_rows(rows, candidate)}
        if candidate_pair_ids == target_pair_ids:
            simplified = candidate
    return simplified


def rule_text(rule: dict):
    parts = []
    low = rule.get("precheck_feature_mean_l2_low")
    high = rule.get("precheck_feature_mean_l2_high")
    if low is not None and high is not None:
        parts.append(
            f"{format_float(low)} <= precheck_feature_mean_l2 <= {format_float(high)}"
        )
    elif low is not None:
        parts.append(f"precheck_feature_mean_l2 >= {format_float(low)}")
    elif high is not None:
        parts.append(f"precheck_feature_mean_l2 <= {format_float(high)}")

    pool_max = rule.get("target_pool_hidden_anomaly_ratio_max")
    if pool_max is not None:
        parts.append(f"target_pool_hidden_anomaly_ratio <= {format_float(pool_max)}")

    val_count_min = rule.get("val_count_min")
    if val_count_min is not None:
        parts.append(f"val_count >= {int(val_count_min)}")

    val_anomaly_count_min = rule.get("val_anomaly_count_min")
    if val_anomaly_count_min is not None:
        parts.append(f"val_anomaly_count >= {int(val_anomaly_count_min)}")

    val_anomaly_ratio_max = rule.get("val_anomaly_ratio_max")
    if val_anomaly_ratio_max is not None:
        parts.append(f"val_anomaly_ratio <= {format_float(val_anomaly_ratio_max)}")

    return " and ".join(parts) if parts else "(no active constraints)"


def write_report(path: Path, summary: dict, selected_rows, all_rows):
    lines = [
        "# SMD Good-Pair Band Selector",
        "",
        "This selector is outcome-guided on the existing SMD top20 benchmark.",
        "A pair is labeled `good` iff both `combined AUROC > uad_source AUROC` and `best NAS AUROC > best fixed AUROC`.",
        "",
        f"- Buildable + benchmarked pairs considered: {summary['n_rows_total']}",
        f"- Good pairs in that set: {summary['n_good_total']}",
        f"- Selected pairs: {summary['selected_stats']['selected_count']}",
        f"- Selected good pairs: {summary['selected_stats']['tp']}",
        f"- Selected bad pairs: {summary['selected_stats']['fp']}",
        f"- Precision: {format_float(summary['selected_stats']['precision'])}",
        f"- Recall over good pairs: {format_float(summary['recall_over_good'])}",
        "",
        "## Selected Rule",
        "",
        f"`{summary['rule_text']}`",
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
        description="Search an outcome-guided SMD pair-selection band over the existing top20 PAD benchmark."
    )
    ap.add_argument(
        "--manifest",
        default=str(PROJ_ROOT / "data" / "smd_experiments" / "cross_entity_top20_pad_prefhard" / "manifest.json"),
    )
    ap.add_argument(
        "--build_summary",
        default=str(PROJ_ROOT / "data" / "smd_experiments" / "cross_entity_top20_pad_prefhard" / "build_summary.json"),
    )
    ap.add_argument(
        "--benchmark_summary",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "smd_top20_pad_prefhard_rerank_compare"
            / "smd_top20_pad_prefhard_rerank"
            / "summary.json"
        ),
    )
    ap.add_argument(
        "--out_dir",
        default=str(PROJ_ROOT / "data" / "smd_experiments" / "cross_entity_top20_goodpair_band"),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    build_summary_path = Path(args.build_summary)
    benchmark_summary_path = Path(args.benchmark_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_json(manifest_path)
    build_summary_rows = read_json(build_summary_path)
    benchmark_rows = read_json(benchmark_summary_path).get("rows") or []

    merged_rows = merge_rows(manifest_rows, build_summary_rows, benchmark_rows)
    if not merged_rows:
        raise RuntimeError("No overlapping buildable + benchmarked SMD rows found.")

    best = search_best_rule(merged_rows)
    simplified_rule = simplify_rule(merged_rows, best["selected_rows"], best["rule"])
    final_selected_rows = select_rows(merged_rows, simplified_rule)
    selected_stats = evaluate_selected(final_selected_rows)
    n_good_total = sum(1 for row in merged_rows if row["is_good"])
    recall_over_good = float(selected_stats["tp"] / n_good_total) if n_good_total else 0.0

    selected_pair_ids = {row["pair_id"] for row in final_selected_rows}
    selected_manifest = []
    for order, row in enumerate(sorted(final_selected_rows, key=lambda item: item["global_pad_rank"]), start=1):
        meta = dict(row["manifest_row"])
        meta["selection_order"] = order
        meta["goodpair_band_selector"] = {
            "rule_text": rule_text(simplified_rule),
            "precheck_feature_mean_l2_low": simplified_rule.get("precheck_feature_mean_l2_low"),
            "precheck_feature_mean_l2_high": simplified_rule.get("precheck_feature_mean_l2_high"),
            "target_pool_hidden_anomaly_ratio_max": simplified_rule.get("target_pool_hidden_anomaly_ratio_max"),
            "val_count_min": simplified_rule.get("val_count_min"),
            "val_anomaly_count_min": simplified_rule.get("val_anomaly_count_min"),
            "val_anomaly_ratio_max": simplified_rule.get("val_anomaly_ratio_max"),
        }
        meta["ranking_rule"] = "cross_entity_top20_goodpair_band"
        selected_manifest.append(meta)

    summary = {
        "selector_name": "cross_entity_top20_goodpair_band",
        "dataset": "smd",
        "source_manifest": str(manifest_path),
        "source_build_summary": str(build_summary_path),
        "source_benchmark_summary": str(benchmark_summary_path),
        "good_definition": {
            "combined_metric": "auroc",
            "nas_metric": "auroc",
            "condition": "combined.auroc > uad_source.auroc and best_nas.auroc > best_fixed.auroc",
        },
        "searched_feature_space": [
            "precheck_feature_mean_l2 interval",
            "target_pool_hidden_anomaly_ratio upper bound",
            "val_count lower bound",
            "val_anomaly_count lower bound",
            "val_anomaly_ratio upper bound",
        ],
        "rule": simplified_rule,
        "rule_text": rule_text(simplified_rule),
        "n_rows_total": int(len(merged_rows)),
        "n_good_total": int(n_good_total),
        "selected_stats": selected_stats,
        "recall_over_good": recall_over_good,
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
        title="SMD Good-Pair Band Selection",
    )
    write_report(out_dir / "REPORT.md", summary, sorted(final_selected_rows, key=lambda item: item["global_pad_rank"]), merged_rows)

    print(f"[DONE] Saved manifest: {out_dir / 'manifest.json'}")
    print(f"[DONE] Selected pairs: {selected_stats['selected_count']} / {len(merged_rows)}")
    print(f"[DONE] Rule: {summary['rule_text']}")


if __name__ == "__main__":
    main()
