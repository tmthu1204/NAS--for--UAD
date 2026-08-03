import argparse
import json
import math
from pathlib import Path


DEFAULT_METRICS = ["auroc", "ap", "f1_best"]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def float_or_none(value):
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def safe_mean(values):
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def fmt(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return ""


def sorted_rows(summary: dict):
    rows = summary.get("rows") or []
    return sorted(
        rows,
        key=lambda row: (
            row.get("global_pad_rank") if row.get("global_pad_rank") is not None else 10**9,
            row.get("pair_id") or "",
        ),
    )


def build_cut_specs(num_rows: int, requested_topk: list[int]):
    specs = []
    for topk in requested_topk:
        if topk * 2 <= num_rows:
            specs.append(
                {
                    "label": f"top{topk}_vs_bottom{topk}",
                    "topk": topk,
                    "bottomk": topk,
                    "middle_excluded": num_rows - (2 * topk),
                }
            )

    half_k = num_rows // 2
    if half_k >= 1 and all(spec["label"] != "top_half_vs_bottom_half" for spec in specs):
        specs.append(
            {
                "label": "top_half_vs_bottom_half",
                "topk": half_k,
                "bottomk": half_k,
                "middle_excluded": num_rows - (2 * half_k),
            }
        )
    return specs


def compute_shift_benefit(rows, metric: str, topk: int, bottomk: int):
    ranked = []
    for row in rows:
        delta = float_or_none((row.get("delta_combined_minus_source") or {}).get(metric))
        ranked.append(
            {
                "pair_id": row.get("pair_id"),
                "pad_rank": row.get("global_pad_rank"),
                "delta": delta,
                "winner_arch": row.get("winner_arch"),
                "source_auroc": float_or_none((row.get("uad_source") or {}).get("auroc")),
                "combined_auroc": float_or_none((row.get("combined") or {}).get("auroc")),
            }
        )

    top_rows = ranked[:topk]
    bottom_rows = ranked[-bottomk:] if bottomk > 0 else []
    top_mean = safe_mean([row["delta"] for row in top_rows])
    bottom_mean = safe_mean([row["delta"] for row in bottom_rows])
    score = None if top_mean is None or bottom_mean is None else float(top_mean - bottom_mean)
    return {
        "metric": metric,
        "topk": topk,
        "bottomk": bottomk,
        "top_mean_delta": top_mean,
        "bottom_mean_delta": bottom_mean,
        "shift_benefit_score": score,
        "top_positive_count": sum(1 for row in top_rows if isinstance(row["delta"], (int, float)) and row["delta"] > 0),
        "bottom_positive_count": sum(1 for row in bottom_rows if isinstance(row["delta"], (int, float)) and row["delta"] > 0),
        "top_rows": top_rows,
        "bottom_rows": bottom_rows,
    }


def build_report(summary_path: Path, summary: dict, metrics: list[str], cut_specs: list[dict]):
    rows = sorted_rows(summary)
    analyses = []
    for metric in metrics:
        for spec in cut_specs:
            analyses.append(
                {
                    "label": spec["label"],
                    "middle_excluded": spec["middle_excluded"],
                    **compute_shift_benefit(rows, metric, spec["topk"], spec["bottomk"]),
                }
            )

    lines = []
    lines.append("# Shift-Benefit Report")
    lines.append("")
    lines.append(f"- Source summary: `{summary_path}`")
    lines.append(f"- Dataset label: `{summary.get('dataset_name', '')}`")
    lines.append(f"- Num pairs: `{len(rows)}`")
    lines.append(
        "- Interpretation: `shift-benefit score = mean(delta_combined-source on top PAD pairs) "
        "- mean(delta_combined-source on low PAD pairs)`."
    )
    lines.append(
        "- Important caveat: this benchmark is already a top-PAD filtered SMD subset, so `low PAD` below means "
        "`relatively lower PAD within the selected benchmark`, not globally low-shift SMD pairs."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Cut | Top mean delta | Bottom mean delta | Shift-benefit score | Top combined>source | Bottom combined>source | Verdict |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in analyses:
        verdict = "supports hypothesis" if (item["shift_benefit_score"] or 0.0) > 0 else "does not support hypothesis"
        lines.append(
            "| "
            + " | ".join(
                [
                    item["metric"],
                    item["label"],
                    fmt(item["top_mean_delta"]),
                    fmt(item["bottom_mean_delta"]),
                    fmt(item["shift_benefit_score"]),
                    str(item["top_positive_count"]),
                    str(item["bottom_positive_count"]),
                    verdict,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Pair-Level Delta By PAD Rank")
    lines.append("")
    lines.append("| PAD rank | Pair | Winner | delta AUROC | delta AUPRC | delta F1_best |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
    for row in rows:
        deltas = row.get("delta_combined_minus_source") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("global_pad_rank", "")),
                    str(row.get("pair_id", "")),
                    str(row.get("winner_arch", "")),
                    fmt(deltas.get("auroc")),
                    fmt(deltas.get("ap")),
                    fmt(deltas.get("f1_best")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Group Members")
    lines.append("")
    for item in analyses:
        lines.append(f"### {item['metric']} - {item['label']}")
        lines.append("")
        lines.append(f"- Top mean delta: `{fmt(item['top_mean_delta'])}`")
        lines.append(f"- Bottom mean delta: `{fmt(item['bottom_mean_delta'])}`")
        lines.append(f"- Shift-benefit score: `{fmt(item['shift_benefit_score'])}`")
        if item["middle_excluded"] > 0:
            lines.append(f"- Middle pairs excluded by this cut: `{item['middle_excluded']}`")
        lines.append("")
        lines.append("| Group | PAD rank | Pair | Delta | Winner | Combined AUROC | Source AUROC |")
        lines.append("| --- | ---: | --- | ---: | --- | ---: | ---: |")
        for group_name, group_rows in [("top", item["top_rows"]), ("bottom", item["bottom_rows"])]:
            for row in group_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            group_name,
                            str(row.get("pad_rank", "")),
                            str(row.get("pair_id", "")),
                            fmt(row.get("delta")),
                            str(row.get("winner_arch", "")),
                            fmt(row.get("combined_auroc")),
                            fmt(row.get("source_auroc")),
                        ]
                    )
                    + " |"
                )
        lines.append("")

    payload = {
        "summary_path": str(summary_path),
        "dataset_name": summary.get("dataset_name"),
        "num_pairs": len(rows),
        "metrics": metrics,
        "cuts": cut_specs,
        "analyses": analyses,
        "rows": rows,
    }
    return payload, "\n".join(lines) + "\n"


def parse_args():
    ap = argparse.ArgumentParser(description="Compute shift-benefit score tables from a manifest benchmark summary.json.")
    ap.add_argument("--summary", required=True, help="Path to summary.json from run_manifest_benchmarks.py")
    ap.add_argument("--metrics", default="auroc,ap,f1_best", help="Comma-separated metrics from delta_combined_minus_source.")
    ap.add_argument("--topk", default="3,5", help="Comma-separated top-k cut sizes. Also adds top-half automatically.")
    ap.add_argument("--out_md", required=True, help="Markdown report path.")
    ap.add_argument("--out_json", default="", help="Optional JSON payload path.")
    return ap.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    summary = read_json(summary_path)
    metrics = [part.strip() for part in args.metrics.split(",") if part.strip()]
    requested_topk = [int(part.strip()) for part in args.topk.split(",") if part.strip()]
    cut_specs = build_cut_specs(len(sorted_rows(summary)), requested_topk)
    payload, report_md = build_report(summary_path, summary, metrics, cut_specs)

    out_md = Path(args.out_md)
    ensure_dir(out_md.parent)
    out_md.write_text(report_md, encoding="utf-8")
    print(f"[DONE] Markdown report: {out_md}")

    if args.out_json.strip():
        out_json = Path(args.out_json)
        ensure_dir(out_json.parent)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[DONE] JSON payload: {out_json}")


if __name__ == "__main__":
    main()
