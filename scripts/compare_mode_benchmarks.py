import argparse
import json
from pathlib import Path


HIGHER_BETTER_METRICS = {
    "ap",
    "auroc",
    "f1_best",
    "f1_pot",
    "event_f1",
    "precision_best",
    "recall_best",
    "precision_pot",
    "recall_pot",
    "event_precision",
    "event_recall",
}

LOWER_BETTER_METRICS = {
    "delay_mean",
    "delay_median",
}


def split_dirs(raw: str):
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def load_benchmark_entries(bench_dirs):
    out = {}
    for bench_dir in bench_dirs:
        if not bench_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {bench_dir}")
        for path in sorted(bench_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            out[path.stem] = {
                "benchmark_dir": bench_dir.name,
                "path": str(path),
                "metrics_uad": payload.get("metrics_uad", {}),
                "search_objective": payload.get("search_objective"),
                "best_arch": payload.get("best_arch"),
            }
    return out


def safe_mean(vals):
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def compare_metric(source_val, combined_val, metric):
    if not isinstance(source_val, (int, float)) or not isinstance(combined_val, (int, float)):
        return None
    if metric in LOWER_BETTER_METRICS:
        if combined_val < source_val:
            return "combined"
        if combined_val > source_val:
            return "source"
        return "tie"
    if combined_val > source_val:
        return "combined"
    if combined_val < source_val:
        return "source"
    return "tie"


def build_summary(source_entries, combined_entries, metrics):
    common_keys = sorted(set(source_entries.keys()) & set(combined_entries.keys()))
    rows = []
    for key in common_keys:
        src = source_entries[key]
        cmb = combined_entries[key]
        deltas = {}
        winners = {}
        for metric in metrics:
            s = src["metrics_uad"].get(metric)
            c = cmb["metrics_uad"].get(metric)
            if isinstance(s, (int, float)) and isinstance(c, (int, float)):
                deltas[metric] = float(c - s)
            winners[metric] = compare_metric(s, c, metric)
        rows.append(
            {
                "case": key,
                "source_benchmark_dir": src["benchmark_dir"],
                "combined_benchmark_dir": cmb["benchmark_dir"],
                "source_metrics_uad": src["metrics_uad"],
                "combined_metrics_uad": cmb["metrics_uad"],
                "delta_combined_minus_source": deltas,
                "winner_by_metric": winners,
            }
        )

    metric_summary = {}
    for metric in metrics:
        source_vals = [row["source_metrics_uad"].get(metric) for row in rows]
        combined_vals = [row["combined_metrics_uad"].get(metric) for row in rows]
        deltas = [row["delta_combined_minus_source"].get(metric) for row in rows]
        winners = [row["winner_by_metric"].get(metric) for row in rows]
        metric_summary[metric] = {
            "source_mean": safe_mean(source_vals),
            "combined_mean": safe_mean(combined_vals),
            "delta_mean": safe_mean(deltas),
            "combined_wins": winners.count("combined"),
            "source_wins": winners.count("source"),
            "ties": winners.count("tie"),
            "num_cases": len(rows),
        }

    return {
        "num_common_cases": len(rows),
        "metrics": metric_summary,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare benchmark outputs from default_nasade uad_source and adaptnas_combined "
            "across the same machine/pair cases."
        )
    )
    ap.add_argument(
        "--source_dirs",
        required=True,
        help="Comma-separated benchmark directories for uad_source results.",
    )
    ap.add_argument(
        "--combined_dirs",
        required=True,
        help="Comma-separated benchmark directories for adaptnas_combined results.",
    )
    ap.add_argument(
        "--metrics",
        default="ap,auroc,f1_best,f1_pot,event_f1,delay_mean",
        help="Comma-separated metric names to compare.",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Optional JSON output path. If omitted, prints summary only.",
    )
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    source_entries = load_benchmark_entries(split_dirs(args.source_dirs))
    combined_entries = load_benchmark_entries(split_dirs(args.combined_dirs))
    summary = build_summary(source_entries, combined_entries, metrics)

    print(f"Common cases: {summary['num_common_cases']}")
    for metric, block in summary["metrics"].items():
        print(
            f"{metric}: "
            f"source_mean={block['source_mean']}, "
            f"combined_mean={block['combined_mean']}, "
            f"delta_mean={block['delta_mean']}, "
            f"combined_wins={block['combined_wins']}/{block['num_cases']}"
        )

    if args.out.strip():
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved comparison: {out_path}")


if __name__ == "__main__":
    main()
