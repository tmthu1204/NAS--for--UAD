import argparse
import json
import math
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.build_pair_rule_common import (  # noqa: E402
    finite_quantile,
    public_rows,
    save_json,
    write_table_csv,
    write_table_markdown,
)
from scripts.select_msl_explainable_paper_safe_rule import (  # noqa: E402
    GOOD_CONDITION_TEXT,
    evaluate_selected,
    format_float,
    format_sci_or_float,
    merge_rows,
    read_json,
)


RULE_NAME = "msl_moderate_shift_proxy_only"


def selected_pair_id(row: dict) -> str:
    return f"{row['source_entity']}__to__{row['target_entity']}"


def finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def compute_thresholds(rows):
    nonperfect = [
        row
        for row in rows
        if isinstance(row.get("domain_auc"), (int, float))
        and math.isfinite(float(row["domain_auc"]))
        and float(row["domain_auc"]) < 1.0
    ]
    if not nonperfect:
        raise RuntimeError("No non-perfect-domain MSL candidates found.")

    return {
        "pad_min": finite_quantile(finite_values(nonperfect, "pad_value"), 0.25),
        "domain_auc_max": finite_quantile(finite_values(nonperfect, "domain_auc"), 0.60),
        "precheck_feature_mean_l2_min": finite_quantile(
            finite_values(rows, "precheck_feature_mean_l2"),
            0.25,
        ),
        "precheck_feature_mean_l2_max": finite_quantile(
            finite_values(nonperfect, "precheck_feature_mean_l2"),
            0.75,
        ),
        "quantiles": {
            "pad_min": "Q25(PAD | domain_auc < 1)",
            "domain_auc_max": "Q60(domain_auc | domain_auc < 1)",
            "precheck_feature_mean_l2_min": "Q25(precheck_feature_mean_l2 | all buildable pairs)",
            "precheck_feature_mean_l2_max": "Q75(precheck_feature_mean_l2 | domain_auc < 1)",
        },
    }


def matches_rule(row: dict, thresholds: dict) -> bool:
    pad = float(row.get("pad_value", float("nan")))
    domain_auc = float(row.get("domain_auc", float("nan")))
    precheck = float(row.get("precheck_feature_mean_l2", float("nan")))
    if not all(math.isfinite(value) for value in [pad, domain_auc, precheck]):
        return False
    return (
        pad >= thresholds["pad_min"]
        and domain_auc <= thresholds["domain_auc_max"]
        and thresholds["precheck_feature_mean_l2_min"]
        <= precheck
        <= thresholds["precheck_feature_mean_l2_max"]
    )


def build_rule_text(thresholds: dict) -> str:
    return (
        "same-prefix buildable; "
        f"PAD >= Q25(PAD | domain_auc < 1) = {format_float(thresholds['pad_min'])}; "
        "TS-JEPA "
        f"domain_auc <= Q60(domain_auc | domain_auc < 1) = {format_float(thresholds['domain_auc_max'])}; "
        f"{format_sci_or_float(thresholds['precheck_feature_mean_l2_min'])} <= "
        "precheck_feature_mean_l2 <= "
        f"{format_sci_or_float(thresholds['precheck_feature_mean_l2_max'])}"
    )


def manifest_rows(selected_rows, rule_text: str, thresholds: dict, stats: dict):
    rows = []
    for order, row in enumerate(selected_rows, start=1):
        meta = dict(row["manifest_row"])
        meta["selection_order"] = order
        meta["selected_shift_level"] = row["shift_level"]
        meta["ranking_rule"] = RULE_NAME
        meta["msl_main_rule"] = {
            "profile": "main",
            "rule_name": RULE_NAME,
            "rule_text": rule_text,
            "thresholds": thresholds,
            "stats": stats,
            "notes": (
                "Proxy-only moderate-shift rule. Thresholds are computed from "
                "pre-benchmark candidate statistics; validation anomaly counts "
                "and test outcomes are not used for selection."
            ),
        }
        rows.append(meta)
    return rows


def summary_rows(rows, selected_rows):
    selected = {row["pair_id"] for row in selected_rows}
    result = []
    for row in rows:
        result.append(
            {
                **row,
                "selected_main": row["pair_id"] in selected,
                "manifest_row": None,
                "ranking_row": None,
                "benchmark_row": None,
            }
        )
    return result


def write_report(path: Path, summary: dict):
    main = summary["profiles"]["main"]
    selected_rows = [row for row in summary["rows"] if row.get("selected_main")]
    thresholds = main["thresholds"]

    lines = [
        "# MSL Moderate-Shift Proxy-Only Main Rule",
        "",
        "This report defines the current main MSL pair-selection rule.",
        "A pair is labeled `good` only for audit/reporting, using "
        f"`{GOOD_CONDITION_TEXT}`.",
        "",
        "## Rule",
        "",
        f"`{main['rule_text']}`",
        "",
        "## Threshold Sources",
        "",
        "| Threshold | Definition | Value |",
        "| --- | --- | ---: |",
        "| PAD min | Q25(PAD | domain_auc < 1) | "
        f"{format_float(thresholds['pad_min'])} |",
        "| domain_auc max | Q60(domain_auc | domain_auc < 1) | "
        f"{format_float(thresholds['domain_auc_max'])} |",
        "| precheck L2 min | Q25(precheck_feature_mean_l2 | all buildable pairs) | "
        f"{format_sci_or_float(thresholds['precheck_feature_mean_l2_min'])} |",
        "| precheck L2 max | Q75(precheck_feature_mean_l2 | domain_auc < 1) | "
        f"{format_sci_or_float(thresholds['precheck_feature_mean_l2_max'])} |",
        "",
        "## Audit Result",
        "",
        f"- Audited MSL pairs considered: {summary['n_rows_total']}",
        f"- Good pairs in audited universe: {summary['n_good_total']}",
        f"- Selected pairs: {main['stats']['selected_count']}",
        f"- True positives: {main['stats']['tp']}",
        f"- False positives: {main['stats']['fp']}",
        f"- Precision: {format_float(main['stats']['precision'])}",
        f"- Recall over good pairs: {format_float(main['stats']['recall_over_good'])}",
        f"- F1 over good pairs: {format_float(main['stats']['f1'])}",
        f"- Mean combined-source AUROC delta: {format_float(main['stats']['mean_delta_combined_minus_source_auroc'])}",
        f"- Mean NAS-fixed AUROC delta: {format_float(main['stats']['mean_delta_nas_minus_fixed_auroc'])}",
        "",
        "## Selected Pairs",
        "",
        "| PAD rank | Prefix | Pair | PAD | Domain AUC | precheck L2 | val anom | d combined-source | d NAS-fixed | good |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in selected_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["buildable_rank"]),
                    row["prefix"],
                    row["pair_id"],
                    format_float(row["pad_value"]),
                    format_float(row["domain_auc"]),
                    format_sci_or_float(row["precheck_feature_mean_l2"]),
                    str(row["val_anomaly_count"]),
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
            "## Paper Explanation",
            "",
            "The rule keeps source-target pairs in a moderate domain-shift band: "
            "the proxy shift is not too weak by PAD, not trivially separable by "
            "TS-JEPA domain AUC, and not an extreme raw-scale outlier by "
            "`precheck_feature_mean_l2`. All thresholds are computed from "
            "pre-benchmark candidate statistics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Select the MSL proxy-only moderate-shift rule as the main rule."
    )
    ap.add_argument(
        "--manifest",
        default=str(
            PROJ_ROOT
            / "data"
            / "msl_experiments"
            / "msl_same_prefix_buildability_auto_20260731"
            / "manifest.json"
        ),
    )
    ap.add_argument(
        "--ranking_json",
        default=str(
            PROJ_ROOT / "outputs" / "benchmarks" / "msl_buildable_tsjepa_pad_rank_20260731.json"
        ),
    )
    ap.add_argument(
        "--benchmark_summary",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "msl_all_prefixes_buildable_paper_51pairs_20260731.json"
        ),
    )
    ap.add_argument(
        "--out_dir",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "msl_moderate_shift_proxy_main_rule_20260804"
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    ranking_path = Path(args.ranking_json)
    benchmark_summary_path = Path(args.benchmark_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_rows(
        read_json(manifest_path),
        read_json(ranking_path),
        read_json(benchmark_summary_path),
    )
    if not rows:
        raise RuntimeError("No overlapping MSL rows found across manifest/ranking/benchmark.")

    thresholds = compute_thresholds(rows)
    selected_rows = sorted(
        [row for row in rows if matches_rule(row, thresholds)],
        key=lambda row: (row["buildable_rank"], row["pair_id"]),
    )
    total_good = sum(1 for row in rows if row["is_good"])
    stats = evaluate_selected(selected_rows, total_good)
    rule_text = build_rule_text(thresholds)
    clean_summary_rows = public_rows(summary_rows(rows, selected_rows))

    summary = {
        "selector_name": "msl_moderate_shift_proxy_main_rule",
        "dataset": "msl",
        "rule_name": RULE_NAME,
        "source_manifest": str(manifest_path),
        "source_ranking_json": str(ranking_path),
        "source_benchmark_summary": str(benchmark_summary_path),
        "good_definition": {
            "combined_metric": "auroc",
            "nas_metric": "auroc",
            "condition": GOOD_CONDITION_TEXT,
        },
        "n_rows_total": len(rows),
        "n_good_total": total_good,
        "profiles": {
            "main": {
                "rule_name": RULE_NAME,
                "rule_text": rule_text,
                "thresholds": thresholds,
                "stats": stats,
                "selected_pairs": [row["pair_id"] for row in selected_rows],
            }
        },
        "rows": clean_summary_rows,
    }

    main_manifest = manifest_rows(selected_rows, rule_text, thresholds, stats)
    save_json(out_dir / "manifest_main.json", main_manifest)
    save_json(out_dir / "manifest.json", main_manifest)
    save_json(out_dir / "selection_summary.json", summary)

    columns = [
        "buildable_rank",
        "prefix",
        "pair_id",
        "pad_value",
        "domain_acc",
        "domain_auc",
        "precheck_feature_mean_l2",
        "val_count",
        "val_anomaly_count",
        "val_anomaly_ratio",
        "delta_combined_minus_source_auroc",
        "delta_nas_minus_fixed_auroc",
        "is_good",
        "selected_main",
    ]
    write_table_csv(out_dir / "selection_summary.csv", clean_summary_rows, columns)
    write_table_markdown(
        out_dir / "selection_summary.md",
        clean_summary_rows,
        columns,
        title="MSL Moderate-Shift Proxy-Only Main Rule",
    )
    write_report(out_dir / "REPORT.md", summary)

    print(f"[DONE] Saved main manifest: {out_dir / 'manifest.json'}")
    print(f"[DONE] Saved explicit main manifest: {out_dir / 'manifest_main.json'}")
    print(f"[DONE] Rule: {rule_text}")
    print(
        "[DONE] Stats: "
        f"selected={stats['selected_count']} tp={stats['tp']} fp={stats['fp']} "
        f"precision={stats['precision']:.4f}"
    )


if __name__ == "__main__":
    main()
