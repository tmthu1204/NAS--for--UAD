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
    public_rows,
    safe_float,
    save_json,
    write_table_csv,
    write_table_markdown,
)


GOOD_CONDITION_TEXT = (
    "combined.auroc > uad_source.auroc and best_NAS.auroc > best_fixed.auroc"
)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def format_float(value, digits=4):
    value = safe_float(value, default=float("nan"))
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def format_sci_or_float(value):
    value = safe_float(value, default=float("nan"))
    if not math.isfinite(value):
        return ""
    if abs(value) >= 1_000_000:
        return f"{value:.1e}"
    return f"{value:.4f}"


def parse_float_or_none_list(raw: str):
    values = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "none":
            values.append(None)
        else:
            values.append(float(token))
    return values


def parse_int_or_none_list(raw: str):
    values = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "none":
            values.append(None)
        else:
            values.append(int(token))
    return values


def parse_bool_list(raw: str):
    values = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"true", "1", "yes"}:
            values.append(True)
        elif token in {"false", "0", "no"}:
            values.append(False)
        else:
            raise ValueError(f"Unsupported boolean token: {item}")
    return values


def row_pair_id(row: dict) -> str:
    return f"{row['source_entity']}__to__{row['target_entity']}"


def merge_rows(manifest_rows, ranking_rows, benchmark_payload):
    manifest_by_pair = {row_pair_id(row): row for row in manifest_rows}
    ranking_by_pair = {row["pair_id"]: row for row in ranking_rows}
    benchmark_rows = benchmark_payload.get("rows") or []
    benchmark_by_pair = {row["pair_id"]: row for row in benchmark_rows}

    merged = []
    for pair_id, manifest_row in manifest_by_pair.items():
        ranking_row = ranking_by_pair.get(pair_id)
        benchmark_row = benchmark_by_pair.get(pair_id)
        if ranking_row is None or benchmark_row is None:
            continue

        val_count = int(manifest_row.get("val_count", 0) or 0)
        val_anom = int(manifest_row.get("val_anomaly_count", 0) or 0)
        test_count = int(manifest_row.get("test_count", 0) or 0)
        test_anom = int(manifest_row.get("test_anomaly_count", 0) or 0)
        pool_count = int(manifest_row.get("target_pool_count", 0) or 0)
        pool_hidden = int(manifest_row.get("target_pool_hidden_anomaly_count", 0) or 0)

        merged.append(
            {
                "pair_id": pair_id,
                "source_entity": manifest_row["source_entity"],
                "target_entity": manifest_row["target_entity"],
                "prefix": str(benchmark_row.get("prefix") or str(manifest_row["source_entity"]).split("-")[0]),
                "shift_level": manifest_row.get("shift_level") or "auto",
                "buildable_rank": int(ranking_row.get("buildable_rank", 10**9)),
                "global_rank": int(ranking_row.get("global_rank", 10**9)),
                "pad_value": safe_float(ranking_row.get("pad_value")),
                "domain_acc": safe_float(ranking_row.get("domain_acc")),
                "domain_auc": safe_float(ranking_row.get("domain_auc")),
                "precheck_feature_mean_l2": safe_float(
                    ((manifest_row.get("candidate_pair_shift_precheck") or {}).get("feature_mean_l2"))
                ),
                "target_pool_count": pool_count,
                "target_pool_hidden_anomaly_count": pool_hidden,
                "target_pool_hidden_anomaly_ratio": safe_float(
                    manifest_row.get("target_pool_hidden_anomaly_ratio")
                ),
                "val_count": val_count,
                "val_anomaly_count": val_anom,
                "val_anomaly_ratio": (
                    float(val_anom / val_count) if val_count > 0 else float("nan")
                ),
                "test_count": test_count,
                "test_anomaly_count": test_anom,
                "test_anomaly_ratio": (
                    float(test_anom / test_count) if test_count > 0 else float("nan")
                ),
                "delta_combined_minus_source_auroc": safe_float(
                    benchmark_row.get("combined_minus_source")
                ),
                "delta_nas_minus_fixed_auroc": safe_float(
                    benchmark_row.get("nas_minus_fixed")
                ),
                "is_good": bool(benchmark_row.get("both")),
                "manifest_row": manifest_row,
                "ranking_row": ranking_row,
                "benchmark_row": benchmark_row,
            }
        )

    merged.sort(key=lambda row: (row["buildable_rank"], row["pair_id"]))
    return merged


def rule_complexity(rule: dict) -> int:
    return sum(
        [
            1 if rule.get("pad_min") is not None else 0,
            1 if rule.get("precheck_max") is not None else 0,
            1 if rule.get("exclude_perfect_domain_auc") else 0,
            1 if rule.get("val_anom_min") is not None else 0,
            1 if rule.get("bucket_b_val_anom_min") is not None else 0,
            1 if rule.get("bucket_a_per_target_cap") is not None else 0,
        ]
    )


def matches_bucket_a(row: dict, rule: dict) -> bool:
    pad_min = rule.get("pad_min")
    if pad_min is not None and safe_float(row.get("pad_value"), default=float("-inf")) < float(pad_min):
        return False

    precheck_max = rule.get("precheck_max")
    precheck = safe_float(row.get("precheck_feature_mean_l2"), default=float("nan"))
    if precheck_max is not None:
        if not math.isfinite(precheck) or precheck > float(precheck_max):
            return False

    if rule.get("exclude_perfect_domain_auc"):
        domain_auc = safe_float(row.get("domain_auc"), default=float("nan"))
        if not math.isfinite(domain_auc) or not (domain_auc < 1.0):
            return False

    val_anom_min = rule.get("val_anom_min")
    if val_anom_min is not None and int(row.get("val_anomaly_count", 0)) < int(val_anom_min):
        return False

    return True


def matches_bucket_b(row: dict, rule: dict) -> bool:
    bucket_b_val_anom_min = rule.get("bucket_b_val_anom_min")
    if bucket_b_val_anom_min is None:
        return False
    return int(row.get("val_anomaly_count", 0)) >= int(bucket_b_val_anom_min)


def select_rows(rows, rule: dict):
    bucket_a_rows = [row for row in rows if matches_bucket_a(row, rule)]
    bucket_a_rows = sorted(bucket_a_rows, key=lambda row: (row["buildable_rank"], row["pair_id"]))

    capped_bucket_a_rows = []
    if rule.get("bucket_a_per_target_cap") is None:
        capped_bucket_a_rows = list(bucket_a_rows)
    else:
        per_target_cap = int(rule["bucket_a_per_target_cap"])
        per_target_counts = {}
        for row in bucket_a_rows:
            target = row["target_entity"]
            used = per_target_counts.get(target, 0)
            if used >= per_target_cap:
                continue
            capped_bucket_a_rows.append(row)
            per_target_counts[target] = used + 1

    selected_by_pair = {row["pair_id"]: row for row in capped_bucket_a_rows}
    bucket_b_rows = [row for row in rows if matches_bucket_b(row, rule)]
    bucket_b_rows = sorted(bucket_b_rows, key=lambda row: (row["buildable_rank"], row["pair_id"]))
    for row in bucket_b_rows:
        selected_by_pair.setdefault(row["pair_id"], row)

    selected_rows = sorted(
        selected_by_pair.values(),
        key=lambda row: (row["buildable_rank"], row["pair_id"]),
    )
    return bucket_a_rows, capped_bucket_a_rows, bucket_b_rows, selected_rows


def evaluate_selected(selected_rows, total_good: int):
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


def mean_value(values):
    finite_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def same_signature(selected_rows):
    return tuple(row["pair_id"] for row in selected_rows)


def candidate_sort_key(candidate: dict):
    stats = candidate["stats"]
    return (
        stats["tp"],
        -stats["fp"],
        stats["precision"],
        stats["recall_over_good"],
        -candidate["complexity"],
    )


def search_candidates(rows, args):
    total_good = sum(1 for row in rows if row["is_good"])
    dedup = {}

    for pad_min in parse_float_or_none_list(args.pad_min_values):
        for precheck_max in parse_float_or_none_list(args.precheck_max_values):
            for exclude_perfect_domain_auc in parse_bool_list(args.exclude_perfect_domain_auc_values):
                for val_anom_min in parse_int_or_none_list(args.val_anom_min_values):
                    for bucket_b_val_anom_min in parse_int_or_none_list(args.bucket_b_val_anom_min_values):
                        for bucket_a_per_target_cap in parse_int_or_none_list(
                            args.bucket_a_per_target_cap_values
                        ):
                            rule = {
                                "pad_min": pad_min,
                                "precheck_max": precheck_max,
                                "exclude_perfect_domain_auc": exclude_perfect_domain_auc,
                                "val_anom_min": val_anom_min,
                                "bucket_b_val_anom_min": bucket_b_val_anom_min,
                                "bucket_a_per_target_cap": bucket_a_per_target_cap,
                            }
                            (
                                bucket_a_rows,
                                capped_bucket_a_rows,
                                bucket_b_rows,
                                selected_rows,
                            ) = select_rows(rows, rule)
                            if not selected_rows:
                                continue

                            stats = evaluate_selected(selected_rows, total_good)
                            candidate = {
                                "rule": rule,
                                "rule_text": build_rule_text(rule),
                                "complexity": rule_complexity(rule),
                                "bucket_a_candidate_count": len(bucket_a_rows),
                                "bucket_a_selected_count": len(capped_bucket_a_rows),
                                "bucket_b_candidate_count": len(bucket_b_rows),
                                "selected_rows": selected_rows,
                                "stats": stats,
                            }
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
    candidates.sort(key=candidate_sort_key, reverse=True)
    return candidates, total_good


def build_rule_text(rule: dict) -> str:
    bucket_a_parts = []
    if rule.get("pad_min") is not None:
        bucket_a_parts.append(f"PAD >= {format_float(rule['pad_min'])}")
    if rule.get("exclude_perfect_domain_auc"):
        bucket_a_parts.append("TS-JEPA domain_auc < 1.0000")
    if rule.get("precheck_max") is not None:
        bucket_a_parts.append(
            f"precheck_feature_mean_l2 <= {format_sci_or_float(rule['precheck_max'])}"
        )
    if rule.get("val_anom_min") is not None:
        bucket_a_parts.append(f"val_anomaly_count >= {int(rule['val_anom_min'])}")
    if not bucket_a_parts:
        bucket_a_parts.append("(no Bucket A filters)")

    text = "Bucket A: " + ", ".join(bucket_a_parts)

    if rule.get("bucket_a_per_target_cap") is not None:
        text += (
            f"; within Bucket A, keep at most {int(rule['bucket_a_per_target_cap'])} "
            "source per target, ordered by buildable PAD rank"
        )

    if rule.get("bucket_b_val_anom_min") is not None:
        text += (
            f". Bucket B override: keep any pair with val_anomaly_count >= "
            f"{int(rule['bucket_b_val_anom_min'])}"
        )

    return text


def choose_balanced_candidate(candidates, args):
    eligible = [
        candidate
        for candidate in candidates
        if candidate["stats"]["precision"] >= args.balanced_min_precision
    ]
    if eligible:
        return max(
            eligible,
            key=lambda candidate: (
                candidate["stats"]["tp"],
                -candidate["stats"]["fp"],
                candidate["stats"]["precision"],
                -candidate["complexity"],
            ),
        )

    return max(
        candidates,
        key=lambda candidate: (
            candidate["stats"]["f1"],
            candidate["stats"]["tp"],
            -candidate["stats"]["fp"],
            candidate["stats"]["precision"],
            -candidate["complexity"],
        ),
    )


def choose_strict_candidate(candidates, args):
    eligible = [
        candidate
        for candidate in candidates
        if candidate["stats"]["fp"] <= args.strict_max_fp
        and candidate["stats"]["tp"] >= args.strict_min_tp
    ]
    if eligible:
        return max(
            eligible,
            key=lambda candidate: (
                candidate["stats"]["precision"],
                candidate["stats"]["tp"],
                -candidate["complexity"],
            ),
        )

    return min(
        candidates,
        key=lambda candidate: (
            candidate["stats"]["fp"],
            -candidate["stats"]["tp"],
            candidate["complexity"],
        ),
    )


def choose_paper_safe_candidate(candidates, args):
    eligible = [
        candidate
        for candidate in candidates
        if candidate["stats"]["fp"] <= args.paper_safe_max_fp
    ]
    if eligible:
        return max(
            eligible,
            key=lambda candidate: (
                candidate["stats"]["tp"],
                -candidate["stats"]["fp"],
                candidate["stats"]["precision"],
                candidate["stats"]["recall_over_good"],
                -candidate["complexity"],
            ),
        )

    return choose_strict_candidate(candidates, args)


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


def candidate_to_manifest_rows(candidate, profile_name: str):
    manifest_rows = []
    for order, row in enumerate(candidate["selected_rows"], start=1):
        meta = dict(row["manifest_row"])
        meta["selection_order"] = order
        meta["selected_shift_level"] = row["shift_level"]
        meta["msl_explainable_paper_safe_rule"] = {
            "profile": profile_name,
            "rule_text": candidate["rule_text"],
            "rule": candidate["rule"],
            "stats": candidate["stats"],
        }
        manifest_rows.append(meta)
    return manifest_rows


def build_summary_rows(rows, strict_candidate, balanced_candidate, paper_safe_candidate):
    strict_pairs = {row["pair_id"] for row in strict_candidate["selected_rows"]}
    balanced_pairs = {row["pair_id"] for row in balanced_candidate["selected_rows"]}
    paper_safe_pairs = {row["pair_id"] for row in paper_safe_candidate["selected_rows"]}
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                **row,
                "selected_strict": row["pair_id"] in strict_pairs,
                "selected_balanced": row["pair_id"] in balanced_pairs,
                "selected_paper_safe": row["pair_id"] in paper_safe_pairs,
                "manifest_row": None,
                "ranking_row": None,
                "benchmark_row": None,
            }
        )
    return summary_rows


def write_report(path: Path, summary: dict):
    rows = summary["rows"]
    strict = summary["profiles"]["strict"]
    balanced = summary["profiles"]["balanced"]
    paper_safe = summary["profiles"]["paper_safe"]
    frontier = summary["frontier"]
    strict_selected_rows = [row for row in rows if row.get("selected_strict")]
    balanced_selected_rows = [row for row in rows if row.get("selected_balanced")]
    paper_safe_selected_rows = [row for row in rows if row.get("selected_paper_safe")]

    lines = [
        "# MSL Explainable Paper-Safe Rule Search",
        "",
        "This report searches simple, explainable rule families over the audited MSL same-prefix buildable universe.",
        "A pair is labeled `good` iff "
        f"`{GOOD_CONDITION_TEXT}`.",
        "",
        f"- Audited MSL pairs considered: {summary['n_rows_total']}",
        f"- Good pairs in the audited universe: {summary['n_good_total']}",
        "",
        "## MSL Feature Note",
        "",
        "- On MSL, `target_pool_hidden_anomaly_ratio` is nearly degenerate and `val_count` is almost constant across pairs.",
        "- The useful discriminators are therefore dominated by `PAD`, TS-JEPA latent `domain_auc`, `precheck_feature_mean_l2`, and `val_anomaly_count`.",
        "",
        "## Recommended Paper-Safe Profile",
        "",
        f"`{paper_safe['rule_text']}`",
        "",
        f"- Selected pairs: {paper_safe['stats']['selected_count']}",
        f"- True positives: {paper_safe['stats']['tp']}",
        f"- False positives: {paper_safe['stats']['fp']}",
        f"- Precision: {format_float(paper_safe['stats']['precision'])}",
        f"- Recall over good pairs: {format_float(paper_safe['stats']['recall_over_good'])}",
        f"- F1 over good pairs: {format_float(paper_safe['stats']['f1'])}",
        "",
        "## Recommended Balanced Profile",
        "",
        f"`{balanced['rule_text']}`",
        "",
        f"- Selected pairs: {balanced['stats']['selected_count']}",
        f"- True positives: {balanced['stats']['tp']}",
        f"- False positives: {balanced['stats']['fp']}",
        f"- Precision: {format_float(balanced['stats']['precision'])}",
        f"- Recall over good pairs: {format_float(balanced['stats']['recall_over_good'])}",
        f"- F1 over good pairs: {format_float(balanced['stats']['f1'])}",
        "",
        "## Strict Zero-FP Profile",
        "",
        f"`{strict['rule_text']}`",
        "",
        f"- Selected pairs: {strict['stats']['selected_count']}",
        f"- True positives: {strict['stats']['tp']}",
        f"- False positives: {strict['stats']['fp']}",
        f"- Precision: {format_float(strict['stats']['precision'])}",
        f"- Recall over good pairs: {format_float(strict['stats']['recall_over_good'])}",
        f"- F1 over good pairs: {format_float(strict['stats']['f1'])}",
        "",
        "## Pareto Frontier",
        "",
        "| Profile | TP | FP | Precision | Recall | Complexity | Rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    frontier_rows = []
    frontier_signatures = set()
    for candidate in frontier:
        signature = (
            candidate["stats"]["tp"],
            candidate["stats"]["fp"],
            round(candidate["stats"]["precision"], 6),
            round(candidate["stats"]["recall_over_good"], 6),
        )
        if signature in frontier_signatures:
            continue
        frontier_signatures.add(signature)
        frontier_rows.append(candidate)
        if len(frontier_rows) >= 12:
            break

    for candidate in frontier_rows:
        label = ""
        if candidate["rule_text"] == paper_safe["rule_text"]:
            label = "paper_safe"
        elif candidate["rule_text"] == balanced["rule_text"]:
            label = "balanced"
        elif candidate["rule_text"] == strict["rule_text"]:
            label = "strict"
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

    def emit_selected_section(title: str, selected_rows):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| PAD rank | Prefix | Pair | PAD | Domain AUC | precheck L2 | val anom | d combined-source | d NAS-fixed | good |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
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

    emit_selected_section("Paper-Safe Selected Pairs", paper_safe_selected_rows)
    emit_selected_section("Balanced Selected Pairs", balanced_selected_rows)
    emit_selected_section("Strict Selected Pairs", strict_selected_rows)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Search explainable paper-safe MSL same-prefix pair-selection rules."
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
            / "msl_explainable_paper_safe_rule_20260731"
        ),
    )
    ap.add_argument("--pad_min_values", default="none,0.5,1.0,1.25,1.5")
    ap.add_argument("--precheck_max_values", default="none,12000000,25000000,30000000")
    ap.add_argument("--exclude_perfect_domain_auc_values", default="false,true")
    ap.add_argument("--val_anom_min_values", default="none,2,3,4")
    ap.add_argument("--bucket_b_val_anom_min_values", default="none,5")
    ap.add_argument("--bucket_a_per_target_cap_values", default="none,1")
    ap.add_argument("--balanced_min_precision", type=float, default=0.70)
    ap.add_argument("--paper_safe_max_fp", type=int, default=1)
    ap.add_argument("--strict_max_fp", type=int, default=0)
    ap.add_argument("--strict_min_tp", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    ranking_path = Path(args.ranking_json)
    benchmark_summary_path = Path(args.benchmark_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_json(manifest_path)
    ranking_rows = read_json(ranking_path)
    benchmark_payload = read_json(benchmark_summary_path)
    rows = merge_rows(manifest_rows, ranking_rows, benchmark_payload)
    if not rows:
        raise RuntimeError("No overlapping MSL rows found across manifest/ranking/benchmark.")

    candidates, total_good = search_candidates(rows, args)
    if not candidates:
        raise RuntimeError("No valid rule candidates were found.")

    strict_candidate = choose_strict_candidate(candidates, args)
    balanced_candidate = choose_balanced_candidate(candidates, args)
    paper_safe_candidate = choose_paper_safe_candidate(candidates, args)
    frontier = build_frontier(candidates)

    summary_rows = build_summary_rows(
        rows,
        strict_candidate,
        balanced_candidate,
        paper_safe_candidate,
    )
    summary = {
        "selector_name": "msl_explainable_paper_safe_rule_search",
        "dataset": "msl",
        "source_manifest": str(manifest_path),
        "source_ranking_json": str(ranking_path),
        "source_benchmark_summary": str(benchmark_summary_path),
        "good_definition": {
            "combined_metric": "auroc",
            "nas_metric": "auroc",
            "condition": GOOD_CONDITION_TEXT,
        },
        "search_space": {
            "pad_min_values": parse_float_or_none_list(args.pad_min_values),
            "precheck_max_values": parse_float_or_none_list(args.precheck_max_values),
            "exclude_perfect_domain_auc_values": parse_bool_list(
                args.exclude_perfect_domain_auc_values
            ),
            "val_anom_min_values": parse_int_or_none_list(args.val_anom_min_values),
            "bucket_b_val_anom_min_values": parse_int_or_none_list(
                args.bucket_b_val_anom_min_values
            ),
            "bucket_a_per_target_cap_values": parse_int_or_none_list(
                args.bucket_a_per_target_cap_values
            ),
        },
        "n_rows_total": len(rows),
        "n_good_total": total_good,
        "profiles": {
            "paper_safe": {
                "rule": paper_safe_candidate["rule"],
                "rule_text": paper_safe_candidate["rule_text"],
                "complexity": paper_safe_candidate["complexity"],
                "stats": paper_safe_candidate["stats"],
                "selected_pairs": [
                    row["pair_id"] for row in paper_safe_candidate["selected_rows"]
                ],
            },
            "strict": {
                "rule": strict_candidate["rule"],
                "rule_text": strict_candidate["rule_text"],
                "complexity": strict_candidate["complexity"],
                "stats": strict_candidate["stats"],
                "selected_pairs": [row["pair_id"] for row in strict_candidate["selected_rows"]],
            },
            "balanced": {
                "rule": balanced_candidate["rule"],
                "rule_text": balanced_candidate["rule_text"],
                "complexity": balanced_candidate["complexity"],
                "stats": balanced_candidate["stats"],
                "selected_pairs": [row["pair_id"] for row in balanced_candidate["selected_rows"]],
            },
        },
        "frontier": [
            {
                "rule": candidate["rule"],
                "rule_text": candidate["rule_text"],
                "complexity": candidate["complexity"],
                "stats": candidate["stats"],
                "selected_pairs": [row["pair_id"] for row in candidate["selected_rows"]],
            }
            for candidate in frontier
        ],
        "rows": public_rows(summary_rows),
    }

    manifest_paper_safe = candidate_to_manifest_rows(paper_safe_candidate, "paper_safe")
    manifest_strict = candidate_to_manifest_rows(strict_candidate, "strict")
    manifest_balanced = candidate_to_manifest_rows(balanced_candidate, "balanced")

    save_json(out_dir / "manifest_paper_safe.json", manifest_paper_safe)
    save_json(out_dir / "manifest_strict.json", manifest_strict)
    save_json(out_dir / "manifest_balanced.json", manifest_balanced)
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
        "selected_paper_safe",
        "selected_strict",
        "selected_balanced",
    ]
    write_table_csv(out_dir / "selection_summary.csv", summary_rows, columns)
    write_table_markdown(
        out_dir / "selection_summary.md",
        summary_rows,
        columns,
        title="MSL Explainable Paper-Safe Rule Search",
    )
    write_report(out_dir / "REPORT.md", summary)

    print(f"[DONE] Saved paper-safe manifest: {out_dir / 'manifest_paper_safe.json'}")
    print(f"[DONE] Saved strict manifest: {out_dir / 'manifest_strict.json'}")
    print(f"[DONE] Saved balanced manifest: {out_dir / 'manifest_balanced.json'}")
    print(f"[DONE] Paper-safe rule: {paper_safe_candidate['rule_text']}")
    print(f"[DONE] Balanced rule: {balanced_candidate['rule_text']}")
    print(f"[DONE] Strict rule: {strict_candidate['rule_text']}")


if __name__ == "__main__":
    main()
