import argparse
import json
import math
from pathlib import Path


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_float(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def finite_values(values):
    out = []
    for value in values:
        value = safe_float(value, default=float("nan"))
        if math.isfinite(value):
            out.append(value)
    return out


def quantile(values, q):
    values = sorted(finite_values(values))
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def load_rows(path: Path):
    payload = read_json(path)
    if isinstance(payload, list):
        return payload, payload
    return payload, payload.get("rows") or []


def find_meta_path(protocol_dir: Path, pair_id: str, shift_level: str):
    direct = protocol_dir / f"{pair_id}__{shift_level}" / "split_metadata.json"
    if direct.exists():
        return direct
    nested = list(protocol_dir.glob(f"**/{pair_id}__{shift_level}/split_metadata.json"))
    if nested:
        return nested[0]
    rank_prefixed = list(protocol_dir.glob(f"**/*__{pair_id}__{shift_level}/split_metadata.json"))
    if rank_prefixed:
        return rank_prefixed[0]
    return None


def enrich_rows(rows, protocol_dir: Path):
    enriched = []
    for row in rows:
        row = dict(row)
        pair_id = row.get("pair_id")
        shift_level = row.get("shift_level") or row.get("selected_shift_level")
        row["shift_level"] = shift_level
        meta_path = None
        if pair_id and shift_level:
            meta_path = find_meta_path(protocol_dir, pair_id, shift_level)
        meta = read_json(meta_path) if meta_path and meta_path.exists() else {}
        if "target_pool_hidden_anomaly_ratio" not in row or row.get("target_pool_hidden_anomaly_ratio") in ("", None):
            row["target_pool_hidden_anomaly_ratio"] = meta.get("target_pool_hidden_anomaly_ratio")
        if "val_count" not in row or row.get("val_count") in ("", None):
            row["val_count"] = meta.get("val_count")
        if "val_anomaly_count" not in row or row.get("val_anomaly_count") in ("", None):
            row["val_anomaly_count"] = meta.get("val_anomaly_count")
        if "pad_value" not in row or row.get("pad_value") in ("", None):
            row["pad_value"] = (meta.get("candidate_pair_shift_precheck") or {}).get("pad_value")
        if "build_success" not in row:
            row["build_success"] = bool(meta)
        val_count = safe_int(row.get("val_count"), default=0)
        val_anom = safe_int(row.get("val_anomaly_count"), default=0)
        if "val_anomaly_ratio" not in row or row.get("val_anomaly_ratio") in ("", None):
            row["val_anomaly_ratio"] = float(val_anom / val_count) if val_count > 0 else float("nan")
        enriched.append(row)
    return enriched


def build_bucket_rule_text(common, bucket_a, bucket_b):
    return (
        "Common filters: "
        f"PAD >= {common['min_pad_value']:.4f}, "
        f"target_pool_hidden_anomaly_ratio <= {common['max_pool_anom_ratio']:.4f}. "
        "Bucket A (val-rich / anomaly-supported): "
        f"val_count >= {bucket_a['min_val_count']}, "
        f"val_anomaly_count >= {bucket_a['min_val_anomaly_count']}, "
        f"val_anomaly_ratio <= {bucket_a['max_val_anomaly_ratio']:.4f}. "
        "Bucket B (compact or lighter-val but still valid): "
        "not in Bucket A, "
        f"val_count >= {bucket_b['min_val_count']}, "
        f"val_anomaly_count >= {bucket_b['min_val_anomaly_count']}, "
        f"val_anomaly_ratio <= {bucket_b['max_val_anomaly_ratio']:.4f}."
    )


def compute_thresholds(rows, args):
    valid_rows = [row for row in rows if row.get("build_success")]
    common = {
        "min_pad_value": max(
            args.min_pad_floor,
            quantile([row.get("pad_value") for row in valid_rows], args.min_pad_quantile),
        ),
        "max_pool_anom_ratio": min(
            args.max_pool_anom_ratio_cap,
            quantile([row.get("target_pool_hidden_anomaly_ratio") for row in valid_rows], args.max_pool_anom_ratio_quantile),
        ),
    }
    bucket_a = {
        "label": args.bucket_a_label,
        "min_val_count": max(
            args.min_val_floor,
            int(math.ceil(quantile([row.get("val_count") for row in valid_rows], args.bucket_a_min_val_count_quantile))),
        ),
        "min_val_anomaly_count": max(
            args.min_val_anom_floor,
            int(math.ceil(quantile([row.get("val_anomaly_count") for row in valid_rows], args.bucket_a_min_val_anom_quantile))),
        ),
        "max_val_anomaly_ratio": min(
            args.bucket_a_max_val_ratio_cap,
            quantile([row.get("val_anomaly_ratio") for row in valid_rows], args.bucket_a_max_val_ratio_quantile),
        ),
    }
    bucket_b = {
        "label": args.bucket_b_label,
        "min_val_count": max(
            args.min_val_floor,
            int(math.ceil(quantile([row.get("val_count") for row in valid_rows], args.bucket_b_min_val_count_quantile))),
        ),
        "min_val_anomaly_count": max(
            args.min_val_anom_floor,
            int(math.ceil(quantile([row.get("val_anomaly_count") for row in valid_rows], args.bucket_b_min_val_anom_quantile))),
        ),
        "max_val_anomaly_ratio": min(
            args.bucket_b_max_val_ratio_cap,
            quantile([row.get("val_anomaly_ratio") for row in valid_rows], args.bucket_b_max_val_ratio_quantile),
        ),
    }
    return {"common": common, "bucket_a": bucket_a, "bucket_b": bucket_b}


def passes_common(row, common):
    if not row.get("build_success"):
        return False
    pad_value = safe_float(row.get("pad_value"))
    pool_ratio = safe_float(row.get("target_pool_hidden_anomaly_ratio"))
    if not math.isfinite(pad_value) or pad_value < common["min_pad_value"]:
        return False
    if not math.isfinite(pool_ratio) or pool_ratio > common["max_pool_anom_ratio"]:
        return False
    return True


def passes_bucket_a(row, thresholds):
    common = thresholds["common"]
    bucket = thresholds["bucket_a"]
    if not passes_common(row, common):
        return False
    val_count = safe_int(row.get("val_count"))
    val_anom = safe_int(row.get("val_anomaly_count"))
    val_ratio = safe_float(row.get("val_anomaly_ratio"))
    if val_count < bucket["min_val_count"]:
        return False
    if val_anom < bucket["min_val_anomaly_count"]:
        return False
    if not math.isfinite(val_ratio) or val_ratio > bucket["max_val_anomaly_ratio"]:
        return False
    return True


def passes_bucket_b(row, thresholds):
    common = thresholds["common"]
    bucket = thresholds["bucket_b"]
    if not passes_common(row, common):
        return False
    if passes_bucket_a(row, thresholds):
        return False
    val_count = safe_int(row.get("val_count"))
    val_anom = safe_int(row.get("val_anomaly_count"))
    val_ratio = safe_float(row.get("val_anomaly_ratio"))
    if val_count < bucket["min_val_count"]:
        return False
    if val_anom < bucket["min_val_anomaly_count"]:
        return False
    if not math.isfinite(val_ratio) or val_ratio > bucket["max_val_anomaly_ratio"]:
        return False
    return True


def rank_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            -safe_float(row.get("pad_value"), default=float("-inf")),
            safe_float(row.get("target_pool_hidden_anomaly_ratio"), default=float("inf")),
            -safe_int(row.get("val_anomaly_count"), default=0),
            -safe_int(row.get("val_count"), default=0),
            safe_int(row.get("global_pad_rank"), default=10**9),
            str(row.get("pair_id", "")),
        ),
    )


def take_with_target_cap(rows, topk, per_target_cap, used_pairs):
    selected = []
    target_counts = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id in used_pairs:
            continue
        target = row.get("target_entity")
        if target_counts.get(target, 0) >= per_target_cap:
            continue
        selected.append(row)
        used_pairs.add(pair_id)
        target_counts[target] = target_counts.get(target, 0) + 1
        if len(selected) >= topk:
            break
    return selected


def delta_metric(row: dict, left_key: str, right_key: str, metric: str):
    left = safe_float(((row.get(left_key) or {}).get(metric)), default=float("nan"))
    right = safe_float(((row.get(right_key) or {}).get(metric)), default=float("nan"))
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return float(left - right)


def audit_selection(selected_rows, benchmark_summary_path: Path | None):
    if benchmark_summary_path is None or not benchmark_summary_path.exists():
        return None
    benchmark_rows = read_json(benchmark_summary_path).get("rows") or []
    benchmark_by_pair = {row["pair_id"]: row for row in benchmark_rows}
    audited = []
    for row in selected_rows:
        pair_id = row["pair_id"]
        benchmark_row = benchmark_by_pair.get(pair_id)
        if benchmark_row is None:
            continue
        d_combined = delta_metric(benchmark_row, "combined", "uad_source", "auroc")
        d_nas = delta_metric(benchmark_row, "nas_bestarch", "best_fixed", "auroc")
        audited.append(
            {
                "pair_id": pair_id,
                "shift_level": row.get("shift_level"),
                "bucket": row.get("selection_bucket"),
                "global_pad_rank": row.get("global_pad_rank"),
                "delta_combined_minus_source_auroc": d_combined,
                "delta_nas_minus_fixed_auroc": d_nas,
                "good": bool(math.isfinite(d_combined) and math.isfinite(d_nas) and d_combined > 0 and d_nas > 0),
            }
        )
    if not audited:
        return None
    good_count = sum(1 for row in audited if row["good"])
    return {
        "audited_pairs": len(audited),
        "good_pairs": good_count,
        "precision": float(good_count / len(audited)),
        "combined_gt_source": sum(1 for row in audited if safe_float(row["delta_combined_minus_source_auroc"]) > 0),
        "nas_gt_fixed": sum(1 for row in audited if safe_float(row["delta_nas_minus_fixed_auroc"]) > 0),
        "rows": audited,
    }


def write_report(path: Path, title: str, thresholds: dict, bucket_a_rows, bucket_b_rows, selected_rows, audit):
    lines = [
        f"# {title}",
        "",
        "## Rule",
        "",
        f"`{build_bucket_rule_text(thresholds['common'], thresholds['bucket_a'], thresholds['bucket_b'])}`",
        "",
        f"- Bucket A candidates: {len(bucket_a_rows)}",
        f"- Bucket B candidates: {len(bucket_b_rows)}",
        f"- Selected pairs: {len(selected_rows)}",
        "",
        "## Selected Pairs",
        "",
        "| Order | Bucket | PAD rank | Pair | Shift | PAD | pool ratio | val count | val anom | val ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(selected_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("selection_bucket", "")),
                    str(row.get("global_pad_rank", "")),
                    str(row.get("pair_id", "")),
                    str(row.get("shift_level", "")),
                    f"{safe_float(row.get('pad_value')):.4f}",
                    f"{safe_float(row.get('target_pool_hidden_anomaly_ratio')):.4f}",
                    str(safe_int(row.get("val_count"))),
                    str(safe_int(row.get("val_anomaly_count"))),
                    f"{safe_float(row.get('val_anomaly_ratio')):.4f}",
                ]
            )
            + " |"
        )
    if audit:
        lines.extend(
            [
                "",
                "## Audit",
                "",
                f"- Audited pairs with benchmark overlap: {audit['audited_pairs']}",
                f"- Good pairs: {audit['good_pairs']}",
                f"- Precision: {audit['precision']:.4f}",
                f"- Combined > Source count: {audit['combined_gt_source']}",
                f"- NAS > Fixed count: {audit['nas_gt_fixed']}",
                "",
                "| Bucket | PAD rank | Pair | d Combined-Source AUROC | d NAS-Fixed AUROC | Good |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in audit["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("bucket", "")),
                        str(row.get("global_pad_rank", "")),
                        str(row.get("pair_id", "")),
                        f"{safe_float(row.get('delta_combined_minus_source_auroc')):.4f}",
                        f"{safe_float(row.get('delta_nas_minus_fixed_auroc')):.4f}",
                        "yes" if row.get("good") else "no",
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Select explainable two-bucket cross-entity pairs from a built candidate universe.")
    ap.add_argument("--selection_rows_json", required=True)
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Explainable Two-Bucket Selection")
    ap.add_argument("--benchmark_summary", default=None)
    ap.add_argument("--min_pad_quantile", type=float, default=0.50)
    ap.add_argument("--max_pool_anom_ratio_quantile", type=float, default=0.90)
    ap.add_argument("--min_pad_floor", type=float, default=1.0)
    ap.add_argument("--max_pool_anom_ratio_cap", type=float, default=0.10)
    ap.add_argument("--min_val_floor", type=int, default=32)
    ap.add_argument("--min_val_anom_floor", type=int, default=7)
    ap.add_argument("--bucket_a_label", default="val_rich")
    ap.add_argument("--bucket_a_min_val_count_quantile", type=float, default=0.30)
    ap.add_argument("--bucket_a_min_val_anom_quantile", type=float, default=0.50)
    ap.add_argument("--bucket_a_max_val_ratio_quantile", type=float, default=0.85)
    ap.add_argument("--bucket_a_max_val_ratio_cap", type=float, default=0.40)
    ap.add_argument("--bucket_b_label", default="compact_or_sparse")
    ap.add_argument("--bucket_b_min_val_count_quantile", type=float, default=0.10)
    ap.add_argument("--bucket_b_min_val_anom_quantile", type=float, default=0.10)
    ap.add_argument("--bucket_b_max_val_ratio_quantile", type=float, default=0.90)
    ap.add_argument("--bucket_b_max_val_ratio_cap", type=float, default=0.50)
    ap.add_argument("--topk_bucket_a", type=int, default=3)
    ap.add_argument("--topk_bucket_b", type=int, default=2)
    ap.add_argument("--per_target_cap", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    selection_rows_path = Path(args.selection_rows_json)
    protocol_dir = Path(args.protocol_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload, rows = load_rows(selection_rows_path)
    rows = enrich_rows(rows, protocol_dir)
    thresholds = compute_thresholds(rows, args)

    bucket_a_rows = [dict(row, selection_bucket=thresholds["bucket_a"]["label"]) for row in rows if passes_bucket_a(row, thresholds)]
    bucket_b_rows = [dict(row, selection_bucket=thresholds["bucket_b"]["label"]) for row in rows if passes_bucket_b(row, thresholds)]
    bucket_a_rows = rank_rows(bucket_a_rows)
    bucket_b_rows = rank_rows(bucket_b_rows)

    used_pairs = set()
    selected_a = take_with_target_cap(bucket_a_rows, args.topk_bucket_a, args.per_target_cap, used_pairs)
    selected_b = take_with_target_cap(bucket_b_rows, args.topk_bucket_b, args.per_target_cap, used_pairs)
    selected_rows = selected_a + selected_b

    manifest = []
    missing = []
    for order, row in enumerate(selected_rows, start=1):
        meta_path = find_meta_path(protocol_dir, row["pair_id"], row["shift_level"])
        if meta_path is None:
            missing.append(f"{row['pair_id']}::{row['shift_level']}")
            continue
        meta = read_json(meta_path)
        meta["selection_order"] = order
        meta["selected_shift_level"] = row["shift_level"]
        meta["two_bucket_selection"] = {
            "selection_bucket": row["selection_bucket"],
            "thresholds": thresholds,
            "rule_text": build_bucket_rule_text(thresholds["common"], thresholds["bucket_a"], thresholds["bucket_b"]),
        }
        manifest.append(meta)

    audit = audit_selection(
        selected_rows,
        Path(args.benchmark_summary) if args.benchmark_summary else None,
    )

    summary = {
        "source_rows_json": str(selection_rows_path),
        "protocol_dir": str(protocol_dir),
        "rule_name": "cross_entity_explainable_two_bucket",
        "thresholds": thresholds,
        "rule_text": build_bucket_rule_text(thresholds["common"], thresholds["bucket_a"], thresholds["bucket_b"]),
        "selected_count": len(selected_rows),
        "bucket_a_candidate_count": len(bucket_a_rows),
        "bucket_b_candidate_count": len(bucket_b_rows),
        "missing_meta_count": len(missing),
        "missing_meta": missing,
        "audit": audit,
    }

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "selection_summary.json", summary)
    write_report(out_dir / "REPORT.md", args.title, thresholds, bucket_a_rows, bucket_b_rows, selected_rows, audit)

    print(f"[DONE] Saved manifest: {out_dir / 'manifest.json'}")
    print(f"[DONE] Selected pairs: {len(selected_rows)}")
    print(f"[DONE] Rule: {summary['rule_text']}")


if __name__ == "__main__":
    main()
