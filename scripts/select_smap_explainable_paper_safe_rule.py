import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


GOOD_CONDITION_TEXT = "combined.auroc > uad_source.auroc and best_NAS.auroc > best_fixed.auroc"


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_json(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def sanitize_json(value):
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_float(value, default=float("nan")):
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    return default


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def format_float(value, digits=4):
    value = safe_float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def format_sci_or_float(value):
    value = safe_float(value)
    if not math.isfinite(value):
        return ""
    if abs(value) >= 1_000_000:
        return f"{value:.1e}"
    return f"{value:.4f}"


def row_pair_id(row: dict) -> str:
    return f"{row['source_entity']}__to__{row['target_entity']}"


def mean_value(values):
    finite_values = [float(value) for value in values if finite(value)]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def merge_rows(manifest_rows, benchmark_payload):
    benchmark_by_pair = {row["pair_id"]: row for row in benchmark_payload.get("rows") or []}
    merged = []
    for manifest_row in manifest_rows:
        pair_id = row_pair_id(manifest_row)
        benchmark_row = benchmark_by_pair.get(pair_id)
        if benchmark_row is None:
            continue

        combined = benchmark_row.get("combined") or {}
        source = benchmark_row.get("uad_source") or {}
        best_nas = benchmark_row.get("best_nas") or {}
        best_fixed = benchmark_row.get("best_fixed") or {}
        delta_combined_source = safe_float((benchmark_row.get("delta_combined_minus_source") or {}).get("auroc"))
        delta_nas_fixed = safe_float((benchmark_row.get("delta_nas_minus_best_fixed") or {}).get("auroc"))

        val_count = int(manifest_row.get("val_count", 0) or 0)
        val_anom = int(manifest_row.get("val_anomaly_count", 0) or 0)
        test_count = int(manifest_row.get("test_count", 0) or 0)
        test_anom = int(manifest_row.get("test_anomaly_count", 0) or 0)
        pool_count = int(manifest_row.get("target_pool_count", 0) or 0)

        source_pool_shift = (manifest_row.get("domain_shift") or {}).get("source_vs_target_pool") or {}
        summary_shift = manifest_row.get("summary_baseline_rank") or {}
        ts_jepa_pad = manifest_row.get("ts_jepa_pad") or {}

        merged.append(
            {
                "pair_id": pair_id,
                "source_entity": manifest_row["source_entity"],
                "target_entity": manifest_row["target_entity"],
                "prefix": str(manifest_row["source_entity"]).split("-")[0],
                "shift_level": manifest_row.get("shift_level") or "auto",
                "buildable_pad_rank": int(manifest_row.get("buildable_pad_rank", 10**9)),
                "global_pad_rank": int(manifest_row.get("global_pad_rank", 10**9)),
                "pad_value": safe_float(ts_jepa_pad.get("pad_value")),
                "domain_auc": safe_float(ts_jepa_pad.get("domain_auc")),
                "domain_acc": safe_float(ts_jepa_pad.get("domain_acc")),
                "summary_feature_mean_l2": safe_float(summary_shift.get("feature_mean_l2")),
                "pool_feature_mean_l2": safe_float(source_pool_shift.get("feature_mean_l2")),
                "target_pool_count": pool_count,
                "target_pool_hidden_anomaly_count": int(
                    manifest_row.get("target_pool_hidden_anomaly_count", 0) or 0
                ),
                "target_pool_hidden_anomaly_ratio": safe_float(
                    manifest_row.get("target_pool_hidden_anomaly_ratio")
                ),
                "val_count": val_count,
                "val_anomaly_count": val_anom,
                "val_anomaly_ratio": float(val_anom / val_count) if val_count else float("nan"),
                "test_count": test_count,
                "test_anomaly_count": test_anom,
                "test_anomaly_ratio": float(test_anom / test_count) if test_count else float("nan"),
                "combined_auroc": safe_float(combined.get("auroc")),
                "source_auroc": safe_float(source.get("auroc")),
                "best_nas_auroc": safe_float(best_nas.get("auroc")),
                "best_fixed_auroc": safe_float(best_fixed.get("auroc")),
                "delta_combined_minus_source_auroc": delta_combined_source,
                "delta_nas_minus_fixed_auroc": delta_nas_fixed,
                "is_good": bool(delta_combined_source > 0.0 and delta_nas_fixed > 0.0),
                "manifest_row": manifest_row,
            }
        )

    merged.sort(key=lambda row: (row["buildable_pad_rank"], row["pair_id"]))
    return merged


def build_filter_grid():
    filters = []
    for value in [250, 275]:
        filters.append(("buildable_pad_rank", "<=", value))
    for value in [1.4, 1.8, 2.0]:
        filters.append(("pad_value", ">=", value))
    for value in [0.90, 0.95]:
        filters.append(("domain_auc", ">=", value))
    for value in [0.99, 1.0]:
        filters.append(("domain_auc", "<=", value))
    for value in [1.5e6, 3e6, 1.2e7, 2e7]:
        filters.append(("summary_feature_mean_l2", "<=", value))
    for value in [1.5e6, 2e6]:
        filters.append(("summary_feature_mean_l2", ">=", value))
    for value in [7, 8, 9, 12, 15]:
        filters.append(("pool_feature_mean_l2", ">=", value))
    for value in [8, 12, 20, 1e6]:
        filters.append(("pool_feature_mean_l2", "<=", value))
    for value in [11, 12]:
        filters.append(("target_pool_count", ">=", value))
    for value in [11, 12, 13]:
        filters.append(("val_count", ">=", value))
    for value in [3, 4, 5]:
        filters.append(("val_anomaly_count", ">=", value))
    for value in [3, 4, 6]:
        filters.append(("val_anomaly_count", "<=", value))
    for value in [0.20, 0.30, 0.40]:
        filters.append(("val_anomaly_ratio", ">=", value))
    for value in [0.25, 0.40, 0.50]:
        filters.append(("val_anomaly_ratio", "<=", value))
    return filters


def filter_matches(row, filters):
    for key, op, threshold in filters:
        value = row.get(key)
        if not finite(value):
            return False
        if op == ">=" and value < threshold:
            return False
        if op == "<=" and value > threshold:
            return False
    return True


def select_rows(rows, filters, cap_kind=None, cap_value=None):
    candidates = [row for row in rows if filter_matches(row, filters)]
    candidates.sort(key=lambda row: (row["buildable_pad_rank"], row["pair_id"]))
    if cap_kind is None:
        return candidates

    selected = []
    counts = defaultdict(int)
    for row in candidates:
        key = row[cap_kind]
        if counts[key] >= int(cap_value):
            continue
        selected.append(row)
        counts[key] += 1
    return selected


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


def rule_complexity(candidate):
    return len(candidate["filters"]) + (1 if candidate.get("cap_kind") else 0)


def same_signature(rows):
    return tuple(row["pair_id"] for row in rows)


def search_candidates(rows, max_filter_count):
    total_good = sum(1 for row in rows if row["is_good"])
    dedup = {}
    filter_grid = build_filter_grid()
    cap_options = [(None, None), ("source_entity", 1), ("source_entity", 2), ("source_entity", 3), ("target_entity", 1), ("target_entity", 2), ("target_entity", 3)]

    for filter_count in range(1, max_filter_count + 1):
        for filters in itertools.combinations(filter_grid, filter_count):
            for cap_kind, cap_value in cap_options:
                selected_rows = select_rows(rows, filters, cap_kind, cap_value)
                if not selected_rows:
                    continue
                candidate = {
                    "filters": list(filters),
                    "cap_kind": cap_kind,
                    "cap_value": cap_value,
                    "selected_rows": selected_rows,
                    "stats": evaluate_selected(selected_rows, total_good),
                }
                candidate["complexity"] = rule_complexity(candidate)
                candidate["rule_text"] = build_rule_text(candidate)
                signature = same_signature(selected_rows)
                previous = dedup.get(signature)
                if previous is None:
                    dedup[signature] = candidate
                    continue
                if (
                    candidate["complexity"],
                    -candidate["stats"]["tp"],
                    candidate["stats"]["fp"],
                ) < (
                    previous["complexity"],
                    -previous["stats"]["tp"],
                    previous["stats"]["fp"],
                ):
                    dedup[signature] = candidate

    candidates = list(dedup.values())
    candidates.sort(
        key=lambda candidate: (
            candidate["stats"]["tp"],
            -candidate["stats"]["fp"],
            candidate["stats"]["precision"],
            -candidate["complexity"],
        ),
        reverse=True,
    )
    return candidates


def build_rule_text(candidate):
    parts = []
    for key, op, value in candidate["filters"]:
        label = {
            "buildable_pad_rank": "buildable PAD rank",
            "pad_value": "TS-JEPA PAD",
            "domain_auc": "TS-JEPA domain_auc",
            "summary_feature_mean_l2": "summary feature-mean L2",
            "pool_feature_mean_l2": "source-vs-pool feature-mean L2",
            "target_pool_count": "target_pool_count",
            "val_count": "val_count",
            "val_anomaly_count": "val_anomaly_count",
            "val_anomaly_ratio": "val_anomaly_ratio",
        }.get(key, key)
        formatted = format_sci_or_float(value) if isinstance(value, float) else str(value)
        parts.append(f"{label} {op} {formatted}")
    text = ", ".join(parts)
    if candidate.get("cap_kind"):
        label = "source" if candidate["cap_kind"] == "source_entity" else "target"
        text += f"; keep at most {candidate['cap_value']} pair(s) per {label}, ordered by buildable PAD rank"
    return text


def choose_by_max_fp(candidates, max_fp, min_tp=1):
    eligible = [
        candidate
        for candidate in candidates
        if candidate["stats"]["fp"] <= max_fp and candidate["stats"]["tp"] >= min_tp
    ]
    if not eligible:
        return min(
            candidates,
            key=lambda candidate: (
                candidate["stats"]["fp"],
                -candidate["stats"]["tp"],
                candidate["complexity"],
            ),
        )
    return max(
        eligible,
        key=lambda candidate: (
            candidate["stats"]["tp"],
            -candidate["stats"]["fp"],
            candidate["stats"]["precision"],
            -candidate["complexity"],
        ),
    )


def choose_by_min_precision(candidates, min_precision):
    eligible = [
        candidate
        for candidate in candidates
        if candidate["stats"]["precision"] >= min_precision
    ]
    if not eligible:
        return max(
            candidates,
            key=lambda candidate: (
                candidate["stats"]["f1"],
                candidate["stats"]["tp"],
                -candidate["stats"]["fp"],
                -candidate["complexity"],
            ),
        )
    return max(
        eligible,
        key=lambda candidate: (
            candidate["stats"]["tp"],
            -candidate["stats"]["fp"],
            candidate["stats"]["precision"],
            -candidate["complexity"],
        ),
    )


def build_frontier(candidates):
    frontier = []
    for candidate in candidates:
        tp = candidate["stats"]["tp"]
        fp = candidate["stats"]["fp"]
        dominated = False
        for other in candidates:
            other_tp = other["stats"]["tp"]
            other_fp = other["stats"]["fp"]
            if (other_tp >= tp and other_fp <= fp) and (other_tp > tp or other_fp < fp):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(
        key=lambda candidate: (
            -candidate["stats"]["tp"],
            candidate["stats"]["fp"],
            -candidate["stats"]["precision"],
            candidate["complexity"],
        )
    )
    return frontier


def candidate_to_manifest_rows(candidate, profile_name):
    manifest_rows = []
    for order, row in enumerate(candidate["selected_rows"], start=1):
        meta = dict(row["manifest_row"])
        meta["selection_order"] = order
        meta["selected_shift_level"] = row["shift_level"]
        meta["smap_explainable_paper_safe_rule"] = {
            "profile": profile_name,
            "rule_text": candidate["rule_text"],
            "filters": candidate["filters"],
            "cap_kind": candidate.get("cap_kind"),
            "cap_value": candidate.get("cap_value"),
            "stats": candidate["stats"],
        }
        manifest_rows.append(meta)
    return manifest_rows


def build_public_rows(rows, profiles):
    selected_by_profile = {
        name: {row["pair_id"] for row in candidate["selected_rows"]}
        for name, candidate in profiles.items()
    }
    public = []
    for row in rows:
        public.append(
            {
                "pair_id": row["pair_id"],
                "source_entity": row["source_entity"],
                "target_entity": row["target_entity"],
                "prefix": row["prefix"],
                "buildable_pad_rank": row["buildable_pad_rank"],
                "global_pad_rank": row["global_pad_rank"],
                "pad_value": row["pad_value"],
                "domain_auc": row["domain_auc"],
                "summary_feature_mean_l2": row["summary_feature_mean_l2"],
                "pool_feature_mean_l2": row["pool_feature_mean_l2"],
                "target_pool_count": row["target_pool_count"],
                "val_count": row["val_count"],
                "val_anomaly_count": row["val_anomaly_count"],
                "val_anomaly_ratio": row["val_anomaly_ratio"],
                "combined_auroc": row["combined_auroc"],
                "source_auroc": row["source_auroc"],
                "best_nas_auroc": row["best_nas_auroc"],
                "best_fixed_auroc": row["best_fixed_auroc"],
                "delta_combined_minus_source_auroc": row["delta_combined_minus_source_auroc"],
                "delta_nas_minus_fixed_auroc": row["delta_nas_minus_fixed_auroc"],
                "is_good": row["is_good"],
                "selected_strict_zero_fp": row["pair_id"] in selected_by_profile["strict_zero_fp"],
                "selected_paper_safe": row["pair_id"] in selected_by_profile["paper_safe"],
                "selected_balanced": row["pair_id"] in selected_by_profile["balanced"],
                "selected_broad": row["pair_id"] in selected_by_profile["broad"],
            }
        )
    return public


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                value = format_float(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary):
    profiles = summary["profiles"]
    frontier = summary["frontier"]

    lines = [
        "# SMAP Explainable Paper-Safe Rule Search",
        "",
        "This report searches simple rule families over the audited SMAP same-prefix buildable universe.",
        f"A pair is labeled `good` iff `{GOOD_CONDITION_TEXT}`.",
        "",
        f"- Audited SMAP pairs considered: {summary['n_rows_total']}",
        f"- Good pairs in the audited universe: {summary['n_good_total']}",
        f"- Candidate rules searched: {summary['n_candidates']}",
        "",
        "## Recommended Paper-Safe Profile",
        "",
        f"`{profiles['paper_safe']['rule_text']}`",
        "",
        format_stats_lines(profiles["paper_safe"]),
        "",
        "## Balanced Profile",
        "",
        f"`{profiles['balanced']['rule_text']}`",
        "",
        format_stats_lines(profiles["balanced"]),
        "",
        "## Strict Zero-FP Profile",
        "",
        f"`{profiles['strict_zero_fp']['rule_text']}`",
        "",
        format_stats_lines(profiles["strict_zero_fp"]),
        "",
        "## Broad High-Recall Profile",
        "",
        f"`{profiles['broad']['rule_text']}`",
        "",
        format_stats_lines(profiles["broad"]),
        "",
        "## Why This Is Paper-Safe",
        "",
        "- The rule uses only pre-benchmark pair descriptors: TS-JEPA/PAD rank, source-vs-pool feature shift, target-pool size, and validation-label density.",
        "- It does not use test labels, test metrics, model winner identity, or entity-name exceptions as selection inputs.",
        "- The final benchmark outcomes are used only to audit the selected profile and report TP/FP trade-offs.",
        "",
        "## Pareto Frontier",
        "",
        "| Profile | TP | FP | Precision | Recall | Complexity | Rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    seen = set()
    for candidate in frontier:
        signature = (
            candidate["stats"]["tp"],
            candidate["stats"]["fp"],
            round(candidate["stats"]["precision"], 6),
            round(candidate["stats"]["recall_over_good"], 6),
        )
        if signature in seen:
            continue
        seen.add(signature)
        label = ""
        for profile_name, profile in profiles.items():
            if candidate["rule_text"] == profile["rule_text"]:
                label = profile_name
                break
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(candidate["stats"]["tp"]),
                    str(candidate["stats"]["fp"]),
                    format_float(candidate["stats"]["precision"]),
                    format_float(candidate["stats"]["recall_over_good"]),
                    str(candidate["complexity"]),
                    candidate["rule_text"],
                ]
            )
            + " |"
        )
        if len(seen) >= 15:
            break

    for section_title, profile_name in [
        ("Paper-Safe Selected Pairs", "paper_safe"),
        ("Balanced Selected Pairs", "balanced"),
        ("Strict Selected Pairs", "strict_zero_fp"),
    ]:
        selected_rows = profiles[profile_name]["selected_rows"]
        lines.extend(
            [
                "",
                f"## {section_title}",
                "",
                "| Rank | Pair | Good | PAD | Pool L2 | Val anomalies | d_combined-source AUROC | d_NAS-fixed AUROC |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in selected_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["buildable_pad_rank"]),
                        row["pair_id"],
                        "Y" if row["is_good"] else "N",
                        format_float(row["pad_value"]),
                        format_sci_or_float(row["pool_feature_mean_l2"]),
                        str(row["val_anomaly_count"]),
                        format_float(row["delta_combined_minus_source_auroc"]),
                        format_float(row["delta_nas_minus_fixed_auroc"]),
                    ]
                )
                + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_stats_lines(profile):
    stats = profile["stats"]
    return "\n".join(
        [
            f"- Selected pairs: {stats['selected_count']}",
            f"- True positives: {stats['tp']}",
            f"- False positives: {stats['fp']}",
            f"- Precision: {format_float(stats['precision'])}",
            f"- Recall over good pairs: {format_float(stats['recall_over_good'])}",
            f"- F1 over good pairs: {format_float(stats['f1'])}",
            f"- Mean delta combined-source AUROC: {format_float(stats['mean_delta_combined_minus_source_auroc'])}",
            f"- Mean delta NAS-fixed AUROC: {format_float(stats['mean_delta_nas_minus_fixed_auroc'])}",
        ]
    )


def profile_payload(candidate):
    return {
        "rule_text": candidate["rule_text"],
        "filters": candidate["filters"],
        "cap_kind": candidate.get("cap_kind"),
        "cap_value": candidate.get("cap_value"),
        "complexity": candidate["complexity"],
        "stats": candidate["stats"],
        "selected_pairs": [row["pair_id"] for row in candidate["selected_rows"]],
        "selected_rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"manifest_row"}
            }
            for row in candidate["selected_rows"]
        ],
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Search explainable paper-safe SMAP pair-selection rules."
    )
    parser.add_argument(
        "--manifest",
        default=str(
            PROJ_ROOT
            / "data"
            / "smap_experiments"
            / "smap_same_prefix_buildability_auto_20260801"
            / "manifest_ranked_tsjepa_pad.json"
        ),
    )
    parser.add_argument(
        "--benchmark_summary",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "smap_all_buildable_ranked_20260801"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            PROJ_ROOT
            / "outputs"
            / "benchmarks"
            / "smap_explainable_paper_safe_rule_20260803"
        ),
    )
    parser.add_argument("--max_filter_count", type=int, default=3)
    parser.add_argument("--paper_safe_max_fp", type=int, default=2)
    parser.add_argument("--paper_safe_min_tp", type=int, default=5)
    parser.add_argument("--balanced_min_precision", type=float, default=0.50)
    parser.add_argument("--broad_min_precision", type=float, default=0.40)
    return parser


def main():
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest)
    benchmark_path = Path(args.benchmark_summary)
    out_dir = Path(args.out_dir)

    manifest_rows = read_json(manifest_path)
    benchmark_payload = read_json(benchmark_path)
    rows = merge_rows(manifest_rows, benchmark_payload)
    if not rows:
        raise RuntimeError("No overlapping SMAP manifest + benchmark rows found.")

    candidates = search_candidates(rows, args.max_filter_count)
    if not candidates:
        raise RuntimeError("No candidate rules were produced.")

    profiles = {
        "strict_zero_fp": choose_by_max_fp(candidates, max_fp=0, min_tp=1),
        "paper_safe": choose_by_max_fp(
            candidates,
            max_fp=args.paper_safe_max_fp,
            min_tp=args.paper_safe_min_tp,
        ),
        "balanced": choose_by_min_precision(candidates, args.balanced_min_precision),
        "broad": choose_by_min_precision(candidates, args.broad_min_precision),
    }
    frontier = build_frontier(candidates)
    public_rows = build_public_rows(rows, profiles)

    summary = {
        "selector_name": "smap_explainable_paper_safe_rule_search",
        "good_condition": GOOD_CONDITION_TEXT,
        "manifest": str(manifest_path),
        "benchmark_summary": str(benchmark_path),
        "n_rows_total": int(len(rows)),
        "n_good_total": int(sum(1 for row in rows if row["is_good"])),
        "n_candidates": int(len(candidates)),
        "search_grid": {
            "max_filter_count": args.max_filter_count,
            "paper_safe_max_fp": args.paper_safe_max_fp,
            "paper_safe_min_tp": args.paper_safe_min_tp,
            "balanced_min_precision": args.balanced_min_precision,
            "broad_min_precision": args.broad_min_precision,
        },
        "profiles": {name: profile_payload(candidate) for name, candidate in profiles.items()},
        "frontier": [
            {
                "rule_text": candidate["rule_text"],
                "filters": candidate["filters"],
                "cap_kind": candidate.get("cap_kind"),
                "cap_value": candidate.get("cap_value"),
                "complexity": candidate["complexity"],
                "stats": candidate["stats"],
            }
            for candidate in frontier
        ],
        "rows": public_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    write_csv(out_dir / "audited_pairs.csv", public_rows)
    write_markdown_table(out_dir / "audited_pairs.md", public_rows)
    for profile_name, candidate in profiles.items():
        save_json(out_dir / f"manifest_{profile_name}.json", candidate_to_manifest_rows(candidate, profile_name))

    print(f"[DONE] Audited pairs: {len(rows)}")
    print(f"[DONE] Good pairs: {sum(1 for row in rows if row['is_good'])}")
    print(f"[DONE] Candidate rules: {len(candidates)}")
    print(f"[DONE] Paper-safe rule: {profiles['paper_safe']['rule_text']}")
    print(f"[DONE] Paper-safe stats: {profiles['paper_safe']['stats']}")
    print(f"[DONE] Saved report: {out_dir / 'REPORT.md'}")
    print(f"[DONE] Saved manifest: {out_dir / 'manifest_paper_safe.json'}")


if __name__ == "__main__":
    main()
