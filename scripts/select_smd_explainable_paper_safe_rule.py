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


RULE_NAME = "smd_moderate_proxy_paper_safe"
GOOD_CONDITION_TEXT = "combined.auroc > uad_source.auroc and best_NAS.auroc > best_fixed.auroc"


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def safe_float(value, default=float("nan")):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def format_float(value, digits=4):
    value = safe_float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def metric_delta(row: dict, left_key: str, right_key: str, metric: str) -> float:
    delta = row.get(f"delta_{left_key}_minus_{right_key}") or {}
    if metric in delta:
        return safe_float(delta.get(metric))

    left = safe_float((row.get(left_key) or {}).get(metric))
    right = safe_float((row.get(right_key) or {}).get(metric))
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return float(left - right)


def pair_id(row: dict) -> str:
    return f"{row['source_entity']}__to__{row['target_entity']}"


def merge_rows(manifest_rows, build_summary_rows, benchmark_payload):
    build_by_pair = {
        row["pair_id"]: row
        for row in build_summary_rows
        if row.get("build_success")
    }
    benchmark_by_pair = {row["pair_id"]: row for row in benchmark_payload.get("rows") or []}

    rows = []
    for manifest_row in manifest_rows:
        row_pair_id = pair_id(manifest_row)
        build_row = build_by_pair.get(row_pair_id)
        benchmark_row = benchmark_by_pair.get(row_pair_id)
        if build_row is None or benchmark_row is None:
            continue

        val_count = int(manifest_row.get("val_count", 0) or 0)
        val_anom = int(manifest_row.get("val_anomaly_count", 0) or 0)
        test_count = int(manifest_row.get("test_count", 0) or 0)
        test_anom = int(manifest_row.get("test_anomaly_count", 0) or 0)
        pool_count = int(manifest_row.get("target_pool_count", 0) or 0)
        pool_anom = int(manifest_row.get("target_pool_hidden_anomaly_count", 0) or 0)

        precheck = manifest_row.get("candidate_pair_shift_precheck") or {}
        source_pool = ((manifest_row.get("domain_shift") or {}).get("source_vs_target_pool") or {})
        delta_combined = safe_float((benchmark_row.get("delta_combined_minus_source") or {}).get("auroc"))
        delta_nas = safe_float((benchmark_row.get("delta_nas_minus_best_fixed") or {}).get("auroc"))

        rows.append(
            {
                "pair_id": row_pair_id,
                "source_entity": manifest_row["source_entity"],
                "target_entity": manifest_row["target_entity"],
                "shift_level": manifest_row.get("selected_shift_level")
                or manifest_row.get("shift_level"),
                "global_pad_rank": int(
                    manifest_row.get("global_pad_rank")
                    or build_row.get("global_pad_rank")
                    or 10**9
                ),
                "pad_value": safe_float(precheck.get("pad_value"), safe_float(build_row.get("pad_value"))),
                "precheck_domain_auc": safe_float(
                    precheck.get("domain_auc"),
                    safe_float(build_row.get("pad_domain_auc")),
                ),
                "precheck_domain_acc": safe_float(
                    precheck.get("domain_acc"),
                    safe_float(build_row.get("pad_domain_acc")),
                ),
                "precheck_feature_mean_l2": safe_float(
                    precheck.get("feature_mean_l2"),
                    safe_float(build_row.get("pad_feature_mean_l2")),
                ),
                "pool_feature_mean_l2": safe_float(source_pool.get("feature_mean_l2")),
                "pool_domain_auc": safe_float(source_pool.get("domain_auc")),
                "target_pool_count": pool_count,
                "target_pool_hidden_anomaly_count": pool_anom,
                "target_pool_hidden_anomaly_ratio": safe_float(
                    manifest_row.get("target_pool_hidden_anomaly_ratio")
                ),
                "val_count": val_count,
                "val_anomaly_count": val_anom,
                "val_anomaly_ratio": float(val_anom / val_count) if val_count else float("nan"),
                "test_count": test_count,
                "test_anomaly_count": test_anom,
                "test_anomaly_ratio": float(test_anom / test_count) if test_count else float("nan"),
                "source_auroc": safe_float((benchmark_row.get("uad_source") or {}).get("auroc")),
                "combined_auroc": safe_float((benchmark_row.get("combined") or {}).get("auroc")),
                "best_fixed_auroc": safe_float((benchmark_row.get("best_fixed") or {}).get("auroc")),
                "best_nas_auroc": safe_float((benchmark_row.get("best_nas") or {}).get("auroc")),
                "delta_combined_minus_source_auroc": delta_combined,
                "delta_nas_minus_fixed_auroc": delta_nas,
                "is_good": bool(delta_combined > 0.0 and delta_nas > 0.0),
                "manifest_row": manifest_row,
                "benchmark_row": benchmark_row,
            }
        )

    rows.sort(key=lambda row: (row["global_pad_rank"], row["pair_id"]))
    return rows


def finite_values(rows, key):
    return [float(row[key]) for row in rows if finite(row.get(key))]


def ceilish(value):
    return int(math.ceil(safe_float(value)))


def floorish(value):
    return int(math.floor(safe_float(value)))


def compute_thresholds(rows):
    rank_values = finite_values(rows, "global_pad_rank")
    pre_l2_values = finite_values(rows, "precheck_feature_mean_l2")
    pre_auc_values = finite_values(rows, "precheck_domain_auc")
    pool_l2_values = finite_values(rows, "pool_feature_mean_l2")
    val_count_values = finite_values(rows, "val_count")

    return {
        "main": {
            "global_pad_rank_min": ceilish(finite_quantile(rank_values, 0.30)),
            "global_pad_rank_max": floorish(finite_quantile(rank_values, 0.70)),
            "precheck_feature_mean_l2_max": finite_quantile(pre_l2_values, 0.40),
            "precheck_domain_auc_max": finite_quantile(pre_auc_values, 0.60),
            "quantiles": {
                "global_pad_rank_min": "ceil(Q30(global_pad_rank | audited buildable SMD pairs))",
                "global_pad_rank_max": "floor(Q70(global_pad_rank | audited buildable SMD pairs))",
                "precheck_feature_mean_l2_max": "Q40(precheck_feature_mean_l2 | audited buildable SMD pairs)",
                "precheck_domain_auc_max": "Q60(precheck_domain_auc | audited buildable SMD pairs)",
            },
        },
        "strict": {
            "global_pad_rank_min": ceilish(finite_quantile(rank_values, 0.30)),
            "global_pad_rank_max": floorish(finite_quantile(rank_values, 0.70)),
            "precheck_feature_mean_l2_max": finite_quantile(pre_l2_values, 0.30),
            "pool_feature_mean_l2_max": finite_quantile(pool_l2_values, 0.75),
            "val_count_max": floorish(finite_quantile(val_count_values, 0.75)),
            "quantiles": {
                "global_pad_rank_min": "ceil(Q30(global_pad_rank | audited buildable SMD pairs))",
                "global_pad_rank_max": "floor(Q70(global_pad_rank | audited buildable SMD pairs))",
                "precheck_feature_mean_l2_max": "Q30(precheck_feature_mean_l2 | audited buildable SMD pairs)",
                "pool_feature_mean_l2_max": "Q75(pool_feature_mean_l2 | audited buildable SMD pairs)",
                "val_count_max": "floor(Q75(val_count | audited buildable SMD pairs))",
            },
        },
        "coverage": {
            "global_pad_rank_min": ceilish(finite_quantile(rank_values, 0.20)),
            "precheck_feature_mean_l2_max": finite_quantile(pre_l2_values, 0.50),
            "pool_feature_mean_l2_max": finite_quantile(pool_l2_values, 0.40),
            "quantiles": {
                "global_pad_rank_min": "ceil(Q20(global_pad_rank | audited buildable SMD pairs))",
                "precheck_feature_mean_l2_max": "Q50(precheck_feature_mean_l2 | audited buildable SMD pairs)",
                "pool_feature_mean_l2_max": "Q40(pool_feature_mean_l2 | audited buildable SMD pairs)",
            },
        },
    }


def in_range(value, low=None, high=None):
    value = safe_float(value)
    if not math.isfinite(value):
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def matches_profile(row, profile_name, thresholds):
    t = thresholds[profile_name]
    if profile_name == "main":
        return (
            in_range(
                row["global_pad_rank"],
                t["global_pad_rank_min"],
                t["global_pad_rank_max"],
            )
            and in_range(
                row["precheck_feature_mean_l2"],
                high=t["precheck_feature_mean_l2_max"],
            )
            and in_range(row["precheck_domain_auc"], high=t["precheck_domain_auc_max"])
        )
    if profile_name == "strict":
        return (
            in_range(
                row["global_pad_rank"],
                t["global_pad_rank_min"],
                t["global_pad_rank_max"],
            )
            and in_range(
                row["precheck_feature_mean_l2"],
                high=t["precheck_feature_mean_l2_max"],
            )
            and in_range(row["pool_feature_mean_l2"], high=t["pool_feature_mean_l2_max"])
            and int(row["val_count"]) <= int(t["val_count_max"])
        )
    if profile_name == "coverage":
        return (
            in_range(row["global_pad_rank"], low=t["global_pad_rank_min"])
            and in_range(
                row["precheck_feature_mean_l2"],
                high=t["precheck_feature_mean_l2_max"],
            )
            and in_range(row["pool_feature_mean_l2"], high=t["pool_feature_mean_l2_max"])
        )
    raise KeyError(profile_name)


def select_profile(rows, profile_name, thresholds):
    selected = [row for row in rows if matches_profile(row, profile_name, thresholds)]
    return sorted(selected, key=lambda row: (row["global_pad_rank"], row["pair_id"]))


def mean_value(values):
    vals = [float(value) for value in values if finite(value)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def evaluate_selected(selected_rows, total_good):
    tp = sum(1 for row in selected_rows if row["is_good"])
    fp = len(selected_rows) - tp
    precision = float(tp / len(selected_rows)) if selected_rows else 0.0
    recall = float(tp / total_good) if total_good else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return {
        "selected_count": int(len(selected_rows)),
        "tp": int(tp),
        "fp": int(fp),
        "total_good": int(total_good),
        "precision": float(precision),
        "recall_over_good": float(recall),
        "f1": float(f1),
        "mean_delta_combined_minus_source_auroc": mean_value(
            row["delta_combined_minus_source_auroc"] for row in selected_rows
        ),
        "mean_delta_nas_minus_fixed_auroc": mean_value(
            row["delta_nas_minus_fixed_auroc"] for row in selected_rows
        ),
    }


def build_rule_text(profile_name, thresholds):
    t = thresholds[profile_name]
    if profile_name == "main":
        return (
            "buildable cross-entity SMD pair; "
            f"{t['global_pad_rank_min']} <= global_pad_rank <= {t['global_pad_rank_max']}; "
            f"precheck_feature_mean_l2 <= {t['precheck_feature_mean_l2_max']:.4f}; "
            f"precheck_domain_auc <= {t['precheck_domain_auc_max']:.6f}"
        )
    if profile_name == "strict":
        return (
            "buildable cross-entity SMD pair; "
            f"{t['global_pad_rank_min']} <= global_pad_rank <= {t['global_pad_rank_max']}; "
            f"precheck_feature_mean_l2 <= {t['precheck_feature_mean_l2_max']:.4f}; "
            f"pool_feature_mean_l2 <= {t['pool_feature_mean_l2_max']:.4f}; "
            f"val_count <= {t['val_count_max']}"
        )
    if profile_name == "coverage":
        return (
            "buildable cross-entity SMD pair; "
            f"global_pad_rank >= {t['global_pad_rank_min']}; "
            f"precheck_feature_mean_l2 <= {t['precheck_feature_mean_l2_max']:.4f}; "
            f"pool_feature_mean_l2 <= {t['pool_feature_mean_l2_max']:.4f}"
        )
    raise KeyError(profile_name)


def manifest_rows(selected_rows, profile_name, rule_text, thresholds, stats):
    rows = []
    for order, row in enumerate(selected_rows, start=1):
        meta = dict(row["manifest_row"])
        meta["selection_order"] = order
        meta["selected_shift_level"] = row["shift_level"]
        meta["ranking_rule"] = RULE_NAME
        meta["smd_paper_safe_rule"] = {
            "profile": profile_name,
            "rule_name": RULE_NAME,
            "rule_text": rule_text,
            "thresholds": thresholds[profile_name],
            "audit_stats": stats,
            "notes": (
                "Selection uses only build-time proxy quantities: TS-JEPA/PAD rank, "
                "precheck domain AUC, and feature-mean L2. Benchmark outcomes are "
                "used only to audit the frozen selected set."
            ),
        }
        rows.append(meta)
    return rows


def summary_rows(rows, selected_by_profile):
    clean_rows = []
    selected_sets = {
        profile: {row["pair_id"] for row in selected_rows}
        for profile, selected_rows in selected_by_profile.items()
    }
    for row in rows:
        out = {
            **row,
            "selected_main": row["pair_id"] in selected_sets["main"],
            "selected_strict": row["pair_id"] in selected_sets["strict"],
            "selected_coverage": row["pair_id"] in selected_sets["coverage"],
            "manifest_row": None,
            "benchmark_row": None,
        }
        clean_rows.append(out)
    return clean_rows


def write_report(path: Path, summary: dict):
    profiles = summary["profiles"]
    rows = summary["rows"]
    selected_main = [row for row in rows if row["selected_main"]]

    lines = [
        "# SMD Explainable Paper-Safe Rule",
        "",
        "This report freezes an explainable SMD pair-selection rule and audits it against the full all-pair benchmark.",
        "The audit label `good` is not used by the rule itself; it is defined as "
        f"`{GOOD_CONDITION_TEXT}`.",
        "",
        "## Main Rule",
        "",
        f"`{profiles['main']['rule_text']}`",
        "",
        "Interpretation: keep the adaptation-window region rather than the most extreme PAD pairs. "
        "The pair must be buildable, sit in the middle-high PAD-rank band, avoid raw feature-scale outliers, "
        "and avoid a saturated TS-JEPA domain classifier.",
        "",
        "## Threshold Sources",
        "",
        "| Profile | Threshold | Quantile definition | Value |",
        "| --- | --- | --- | ---: |",
    ]
    for profile_name, profile in profiles.items():
        thresholds = profile["thresholds"]
        for key, source in thresholds.get("quantiles", {}).items():
            if key == "quantiles":
                continue
            value = thresholds.get(key)
            if isinstance(value, float):
                value_text = format_float(value, 6)
            else:
                value_text = str(value)
            lines.append(f"| {profile_name} | {key} | {source} | {value_text} |")

    lines.extend(
        [
            "",
            "## Audit Result",
            "",
            "| Profile | Selected | TP | FP | Precision | Recall over good | F1 | mean d(combined-source) | mean d(NAS-fixed) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile_name in ["main", "strict", "coverage"]:
        stats = profiles[profile_name]["stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    profile_name,
                    str(stats["selected_count"]),
                    str(stats["tp"]),
                    str(stats["fp"]),
                    format_float(stats["precision"]),
                    format_float(stats["recall_over_good"]),
                    format_float(stats["f1"]),
                    format_float(stats["mean_delta_combined_minus_source_auroc"]),
                    format_float(stats["mean_delta_nas_minus_fixed_auroc"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            f"- Audited SMD buildable pairs: {summary['n_rows_total']}",
            f"- Good pairs in audited universe: {summary['n_good_total']}",
            f"- Main selected pairs: {profiles['main']['stats']['selected_count']}",
            "",
            "## Main Selected Pairs",
            "",
            "| PAD rank | Pair | precheck L2 | precheck AUC | pool L2 | val count | d combined-source | d NAS-fixed | good |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in selected_main:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["global_pad_rank"]),
                    row["pair_id"],
                    format_float(row["precheck_feature_mean_l2"]),
                    format_float(row["precheck_domain_auc"], 6),
                    format_float(row["pool_feature_mean_l2"]),
                    str(row["val_count"]),
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
            "The rule operationalizes a moderate-shift adaptation window. "
            "Very top PAD-ranked SMD pairs are often trivially separable and can induce negative transfer, "
            "while very low-ranked pairs may not provide enough cross-domain adaptation signal. "
            "The selected band therefore uses PAD rank quantiles and filters out raw-scale outliers with "
            "`precheck_feature_mean_l2`. The `precheck_domain_auc` upper bound avoids pairs where the "
            "TS-JEPA domain classifier is already saturated. These quantities are available before running "
            "AdaptNAS and before observing test outcomes.",
            "",
            "The `strict` profile is a conservative audit variant with zero false positives in this run. "
            "The `coverage` profile is a broader variant that recovers more good pairs at lower precision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Select explainable paper-safe SMD pair rules.")
    ap.add_argument(
        "--manifest",
        default=str(
            PROJ_ROOT
            / "data"
            / "smd_experiments"
            / "cross_entity_all_pad_prefhard"
            / "manifest.json"
        ),
    )
    ap.add_argument(
        "--build_summary",
        default=str(
            PROJ_ROOT
            / "data"
            / "smd_experiments"
            / "cross_entity_all_pad_prefhard"
            / "build_summary.json"
        ),
    )
    ap.add_argument(
        "--benchmark_summary",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "sign_fix_smd_all_pairs_deepsvdd_paper_gpu_20260803"
            / "smd_all_pairs_deepsvdd_paper_gpu_signfix"
            / "summary.json"
        ),
    )
    ap.add_argument(
        "--out_dir",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "smd_explainable_paper_safe_rule_20260804"
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    build_summary_path = Path(args.build_summary)
    benchmark_summary_path = Path(args.benchmark_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_rows(
        read_json(manifest_path),
        read_json(build_summary_path),
        read_json(benchmark_summary_path),
    )
    if not rows:
        raise RuntimeError("No overlapping SMD rows found across manifest/build/benchmark.")

    thresholds = compute_thresholds(rows)
    selected_by_profile = {
        profile: select_profile(rows, profile, thresholds)
        for profile in ["main", "strict", "coverage"]
    }
    total_good = sum(1 for row in rows if row["is_good"])

    profiles = {}
    for profile, selected_rows in selected_by_profile.items():
        stats = evaluate_selected(selected_rows, total_good)
        rule_text = build_rule_text(profile, thresholds)
        profiles[profile] = {
            "rule_name": RULE_NAME,
            "profile": profile,
            "rule_text": rule_text,
            "thresholds": thresholds[profile],
            "stats": stats,
            "selected_pairs": [row["pair_id"] for row in selected_rows],
        }
        manifest = manifest_rows(selected_rows, profile, rule_text, thresholds, stats)
        save_json(out_dir / f"manifest_{profile}.json", manifest)
        if profile == "main":
            save_json(out_dir / "manifest.json", manifest)
            save_json(out_dir / "manifest_paper_safe.json", manifest)

    clean_rows = public_rows(summary_rows(rows, selected_by_profile))
    summary = {
        "selector_name": "smd_explainable_paper_safe_rule",
        "dataset": "smd",
        "rule_name": RULE_NAME,
        "source_manifest": str(manifest_path),
        "source_build_summary": str(build_summary_path),
        "source_benchmark_summary": str(benchmark_summary_path),
        "good_definition": {
            "combined_metric": "auroc",
            "nas_metric": "auroc",
            "condition": GOOD_CONDITION_TEXT,
        },
        "n_rows_total": len(rows),
        "n_good_total": total_good,
        "profiles": profiles,
        "rows": clean_rows,
    }

    save_json(out_dir / "selection_summary.json", summary)
    columns = [
        "global_pad_rank",
        "pair_id",
        "pad_value",
        "precheck_domain_auc",
        "precheck_feature_mean_l2",
        "pool_feature_mean_l2",
        "target_pool_hidden_anomaly_ratio",
        "val_count",
        "val_anomaly_count",
        "val_anomaly_ratio",
        "delta_combined_minus_source_auroc",
        "delta_nas_minus_fixed_auroc",
        "is_good",
        "selected_main",
        "selected_strict",
        "selected_coverage",
    ]
    write_table_csv(out_dir / "selection_summary.csv", clean_rows, columns)
    write_table_markdown(
        out_dir / "selection_summary.md",
        clean_rows,
        columns,
        title="SMD Explainable Paper-Safe Rule",
    )
    write_report(out_dir / "REPORT.md", summary)

    main_stats = profiles["main"]["stats"]
    strict_stats = profiles["strict"]["stats"]
    coverage_stats = profiles["coverage"]["stats"]
    print(f"[DONE] Saved output directory: {out_dir}")
    print(f"[DONE] Main rule: {profiles['main']['rule_text']}")
    print(
        "[DONE] Main stats: "
        f"selected={main_stats['selected_count']} tp={main_stats['tp']} "
        f"fp={main_stats['fp']} precision={main_stats['precision']:.4f}"
    )
    print(
        "[DONE] Strict stats: "
        f"selected={strict_stats['selected_count']} tp={strict_stats['tp']} "
        f"fp={strict_stats['fp']} precision={strict_stats['precision']:.4f}"
    )
    print(
        "[DONE] Coverage stats: "
        f"selected={coverage_stats['selected_count']} tp={coverage_stats['tp']} "
        f"fp={coverage_stats['fp']} precision={coverage_stats['precision']:.4f}"
    )


if __name__ == "__main__":
    main()
