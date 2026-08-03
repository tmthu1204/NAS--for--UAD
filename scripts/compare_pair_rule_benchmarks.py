import argparse
import json
import math
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def float_or_none(value):
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def mean_metric(rows, mode_key: str, metric: str):
    values = []
    for row in rows:
        value = float_or_none((row.get(mode_key) or {}).get(metric))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def count_true(rows, predicate):
    return sum(1 for row in rows if predicate(row))


def fmt(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return ""


def is_nas_arch(arch_name):
    return isinstance(arch_name, str) and arch_name.startswith("NAS_")


def summarize(summary: dict):
    rows = summary.get("rows") or []
    return {
        "pair_count": len(rows),
        "source_auroc": mean_metric(rows, "uad_source", "auroc"),
        "source_auprc": mean_metric(rows, "uad_source", "ap"),
        "combined_auroc": mean_metric(rows, "combined", "auroc"),
        "combined_auprc": mean_metric(rows, "combined", "ap"),
        "nas_auroc": mean_metric(rows, "nas_bestarch", "auroc"),
        "nas_auprc": mean_metric(rows, "nas_bestarch", "ap"),
        "fixed_auroc": mean_metric(rows, "best_fixed", "auroc"),
        "fixed_auprc": mean_metric(rows, "best_fixed", "ap"),
        "combined_gt_source": count_true(
            rows,
            lambda row: float_or_none((row.get("delta_combined_minus_source") or {}).get("auroc")) not in (None,)
            and float((row.get("delta_combined_minus_source") or {}).get("auroc")) > 0,
        ),
        "nas_gt_fixed": count_true(
            rows,
            lambda row: float_or_none((row.get("delta_nas_minus_best_fixed") or {}).get("auroc")) not in (None,)
            and float((row.get("delta_nas_minus_best_fixed") or {}).get("auroc")) > 0,
        ),
        "winner_is_nas": count_true(rows, lambda row: is_nas_arch(row.get("winner_arch"))),
        "rows": rows,
    }


def write_report(output_path: Path, dataset_summaries: dict, *, old_label: str, new_label: str):
    lines = ["# Pair Rule Benchmark Comparison", ""]
    lines.append("| Dataset | Rule | Pairs | Source AUROC | Combined AUROC | NAS AUROC | Best fixed AUROC | Source AUPRC | Combined AUPRC | NAS AUPRC | Best fixed AUPRC | Combined > Source | NAS > Fixed | Winner is NAS |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset, rules in dataset_summaries.items():
        for rule_name, summary in rules.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        rule_name,
                        str(summary["pair_count"]),
                        fmt(summary["source_auroc"]),
                        fmt(summary["combined_auroc"]),
                        fmt(summary["nas_auroc"]),
                        fmt(summary["fixed_auroc"]),
                        fmt(summary["source_auprc"]),
                        fmt(summary["combined_auprc"]),
                        fmt(summary["nas_auprc"]),
                        fmt(summary["fixed_auprc"]),
                        str(summary["combined_gt_source"]),
                        str(summary["nas_gt_fixed"]),
                        str(summary["winner_is_nas"]),
                    ]
                )
                + " |"
            )

    for dataset, rules in dataset_summaries.items():
        old = rules.get(old_label)
        new = rules.get(new_label)
        if old is None or new is None:
            continue
        lines.extend(["", f"## {dataset.upper()} Delta: new - old", ""])
        lines.append("| Metric | Delta |")
        lines.append("| --- | ---: |")
        lines.append(f"| Combined AUROC | {fmt((new['combined_auroc'] or 0.0) - (old['combined_auroc'] or 0.0))} |")
        lines.append(f"| NAS AUROC | {fmt((new['nas_auroc'] or 0.0) - (old['nas_auroc'] or 0.0))} |")
        lines.append(f"| Best fixed AUROC | {fmt((new['fixed_auroc'] or 0.0) - (old['fixed_auroc'] or 0.0))} |")
        lines.append(f"| Combined > Source count | {new['combined_gt_source'] - old['combined_gt_source']} |")
        lines.append(f"| NAS > Fixed count | {new['nas_gt_fixed'] - old['nas_gt_fixed']} |")
        lines.append(f"| Winner is NAS count | {new['winner_is_nas'] - old['winner_is_nas']} |")

        lines.extend(["", f"### {dataset.upper()} {new_label} Pairs", ""])
        lines.append("| Pair | Shift | Score | PAD rank | PAD | Winner | Combined AUROC | Source AUROC | NAS AUROC | Fixed AUROC | d Combined-Source | d NAS-Fixed |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        new_rows = sorted(
            new["rows"],
            key=lambda row: (
                float_or_none(row.get("rule_selection_score")) or -1.0,
                -(row.get("global_pad_rank") or 10**9),
            ),
            reverse=True,
        )
        for row in new_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("pair_id", ""),
                        str(row.get("shift_level", "")),
                        fmt(row.get("rule_selection_score")),
                        str(row.get("global_pad_rank", "")),
                        fmt(row.get("pad_value")),
                        str(row.get("winner_arch", "")),
                        fmt((row.get("combined") or {}).get("auroc")),
                        fmt((row.get("uad_source") or {}).get("auroc")),
                        fmt((row.get("nas_bestarch") or {}).get("auroc")),
                        fmt((row.get("best_fixed") or {}).get("auroc")),
                        fmt((row.get("delta_combined_minus_source") or {}).get("auroc")),
                        fmt((row.get("delta_nas_minus_best_fixed") or {}).get("auroc")),
                    ]
                )
                + " |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Compare benchmark outputs across pair-selection rules.")
    ap.add_argument("--old_root", required=True, help="Old benchmark root, e.g. outputs/benchmarks/cross_entity_hard_topPAD_global_top3_compare")
    ap.add_argument("--new_root", required=True, help="New benchmark root, e.g. outputs/benchmarks/cross_entity_learnable_shift_val_rich_compare")
    ap.add_argument("--datasets", default="smd,hai")
    ap.add_argument("--old_label", default="topPAD_hard")
    ap.add_argument("--new_label", default="learnable_shift_val_rich")
    ap.add_argument("--out", required=True, help="Markdown report path.")
    args = ap.parse_args()

    old_root = Path(args.old_root)
    new_root = Path(args.new_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    dataset_summaries = {}
    for dataset in datasets:
        rules = {}
        old_summary_path = old_root / dataset / "summary.json"
        new_summary_path = new_root / dataset / "summary.json"
        if old_summary_path.exists():
            rules[args.old_label] = summarize(read_json(old_summary_path))
        if new_summary_path.exists():
            rules[args.new_label] = summarize(read_json(new_summary_path))
        dataset_summaries[dataset] = rules

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(out_path, dataset_summaries, old_label=args.old_label, new_label=args.new_label)


if __name__ == "__main__":
    main()
