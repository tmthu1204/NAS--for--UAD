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
        if "pad_feature_mean_l2" not in row or row.get("pad_feature_mean_l2") in ("", None):
            row["pad_feature_mean_l2"] = (meta.get("candidate_pair_shift_precheck") or {}).get("feature_mean_l2")
        if "build_success" not in row:
            row["build_success"] = bool(meta)
        val_count = safe_int(row.get("val_count"), default=0)
        val_anom = safe_int(row.get("val_anomaly_count"), default=0)
        if "val_anomaly_ratio" not in row or row.get("val_anomaly_ratio") in ("", None):
            row["val_anomaly_ratio"] = float(val_anom / val_count) if val_count > 0 else float("nan")
        enriched.append(row)
    return enriched


def delta_metric(row: dict, left_key: str, right_key: str, metric: str):
    left = safe_float(((row.get(left_key) or {}).get(metric)), default=float("nan"))
    right = safe_float(((row.get(right_key) or {}).get(metric)), default=float("nan"))
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return float(left - right)


def build_rule_text(thresholds: dict):
    return (
        f"PAD >= {thresholds['min_pad_value']:.4f}, "
        f"{thresholds['min_precheck_l2']:.4f} <= precheck_feature_mean_l2 <= {thresholds['max_precheck_l2']:.4f}, "
        f"target_pool_hidden_anomaly_ratio <= {thresholds['max_pool_anom_ratio']:.4f}, "
        f"val_count >= {thresholds['min_val_count']}, "
        f"val_anomaly_count >= {thresholds['min_val_anomaly_count']}, "
        f"val_anomaly_ratio <= {thresholds['max_val_anomaly_ratio']:.4f}"
    )


def select_rows(rows, thresholds):
    selected = []
    for row in rows:
        if not row.get("build_success"):
            continue
        pad_value = safe_float(row.get("pad_value"))
        precheck = safe_float(row.get("pad_feature_mean_l2"))
        pool_ratio = safe_float(row.get("target_pool_hidden_anomaly_ratio"))
        val_count = safe_int(row.get("val_count"))
        val_anom = safe_int(row.get("val_anomaly_count"))
        val_ratio = safe_float(row.get("val_anomaly_ratio"))

        if not math.isfinite(pad_value) or pad_value < thresholds["min_pad_value"]:
            continue
        if not math.isfinite(precheck):
            continue
        if precheck < thresholds["min_precheck_l2"] or precheck > thresholds["max_precheck_l2"]:
            continue
        if not math.isfinite(pool_ratio) or pool_ratio > thresholds["max_pool_anom_ratio"]:
            continue
        if val_count < thresholds["min_val_count"]:
            continue
        if val_anom < thresholds["min_val_anomaly_count"]:
            continue
        if not math.isfinite(val_ratio) or val_ratio > thresholds["max_val_anomaly_ratio"]:
            continue
        selected.append(row)
    return selected


def compute_thresholds(rows, args):
    valid_rows = [row for row in rows if row.get("build_success")]
    precheck_values = [row.get("pad_feature_mean_l2") for row in valid_rows]
    if args.precheck_band_mode == "tail_adaptive":
        q10 = quantile(precheck_values, 0.10)
        q50 = quantile(precheck_values, 0.50)
        q90 = quantile(precheck_values, 0.90)
        left_span = q50 - q10 if math.isfinite(q10) and math.isfinite(q50) else float("nan")
        right_span = q90 - q50 if math.isfinite(q50) and math.isfinite(q90) else float("nan")
        tail_ratio = (
            float(right_span / left_span)
            if math.isfinite(left_span) and left_span > 0 and math.isfinite(right_span)
            else float("nan")
        )
        if math.isfinite(tail_ratio) and tail_ratio >= args.tail_ratio_threshold:
            precheck_low_quantile = args.right_tail_precheck_low_quantile
            precheck_high_quantile = args.right_tail_precheck_high_quantile
            precheck_band_label = "right_tail_midband"
        else:
            precheck_low_quantile = args.compact_precheck_low_quantile
            precheck_high_quantile = args.compact_precheck_high_quantile
            precheck_band_label = "compact_upperhalf"
    else:
        tail_ratio = float("nan")
        precheck_low_quantile = args.precheck_low_quantile
        precheck_high_quantile = args.precheck_high_quantile
        precheck_band_label = "fixed_quantiles"

    thresholds = {
        "min_pad_value": max(args.min_pad_floor, quantile([row.get("pad_value") for row in valid_rows], args.min_pad_quantile)),
        "min_precheck_l2": quantile(precheck_values, precheck_low_quantile),
        "max_precheck_l2": quantile(precheck_values, precheck_high_quantile),
        "max_pool_anom_ratio": min(
            args.max_pool_anom_ratio_cap,
            quantile([row.get("target_pool_hidden_anomaly_ratio") for row in valid_rows], args.max_pool_anom_ratio_quantile),
        ),
        "min_val_count": max(
            args.min_val_floor,
            int(math.ceil(quantile([row.get("val_count") for row in valid_rows], args.min_val_count_quantile))),
        ),
        "min_val_anomaly_count": max(
            args.min_val_anom_floor,
            int(math.ceil(quantile([row.get("val_anomaly_count") for row in valid_rows], args.min_val_anom_quantile))),
        ),
        "max_val_anomaly_ratio": min(
            args.max_val_anom_ratio_cap,
            quantile([row.get("val_anomaly_ratio") for row in valid_rows], args.max_val_anom_ratio_quantile),
        ),
        "precheck_band_mode": args.precheck_band_mode,
        "precheck_band_label": precheck_band_label,
        "precheck_low_quantile_used": precheck_low_quantile,
        "precheck_high_quantile_used": precheck_high_quantile,
        "precheck_tail_ratio_q90_q50_over_q50_q10": tail_ratio,
    }
    return thresholds


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


def write_report(path: Path, title: str, thresholds: dict, selected_rows, audit):
    lines = [
        f"# {title}",
        "",
        "## Rule",
        "",
        f"`{build_rule_text(thresholds)}`",
        "",
        f"- Selected pairs: {len(selected_rows)}",
    ]
    if audit:
        lines.extend(
            [
                f"- Audited pairs with benchmark overlap: {audit['audited_pairs']}",
                f"- Good pairs: {audit['good_pairs']}",
                f"- Precision: {audit['precision']:.4f}",
                f"- Combined > Source count: {audit['combined_gt_source']}",
                f"- NAS > Fixed count: {audit['nas_gt_fixed']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Selected Pairs",
            "",
            "| PAD rank | Pair | Shift | PAD | precheck L2 | pool ratio | val count | val anom | val ratio |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in selected_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("global_pad_rank", "")),
                    str(row.get("pair_id", "")),
                    str(row.get("shift_level", "")),
                    f"{safe_float(row.get('pad_value')):.4f}",
                    f"{safe_float(row.get('pad_feature_mean_l2')):.4f}",
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
                "| PAD rank | Pair | Shift | d Combined-Source AUROC | d NAS-Fixed AUROC | Good |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in audit["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("global_pad_rank", "")),
                        str(row.get("pair_id", "")),
                        str(row.get("shift_level", "")),
                        f"{safe_float(row.get('delta_combined_minus_source_auroc')):.4f}",
                        f"{safe_float(row.get('delta_nas_minus_fixed_auroc')):.4f}",
                        "yes" if row.get("good") else "no",
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Select pairs with an explainable cross-entity hard learnable mid-band rule.")
    ap.add_argument("--selection_rows_json", required=True)
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Explainable Mid-Band Selection")
    ap.add_argument("--benchmark_summary", default=None)
    ap.add_argument("--min_pad_quantile", type=float, default=0.50)
    ap.add_argument("--precheck_band_mode", choices=["fixed_quantiles", "tail_adaptive"], default="fixed_quantiles")
    ap.add_argument("--precheck_low_quantile", type=float, default=0.10)
    ap.add_argument("--precheck_high_quantile", type=float, default=0.55)
    ap.add_argument("--tail_ratio_threshold", type=float, default=1.0)
    ap.add_argument("--right_tail_precheck_low_quantile", type=float, default=0.10)
    ap.add_argument("--right_tail_precheck_high_quantile", type=float, default=0.55)
    ap.add_argument("--compact_precheck_low_quantile", type=float, default=0.50)
    ap.add_argument("--compact_precheck_high_quantile", type=float, default=0.90)
    ap.add_argument("--max_pool_anom_ratio_quantile", type=float, default=0.80)
    ap.add_argument("--min_val_count_quantile", type=float, default=0.30)
    ap.add_argument("--min_val_anom_quantile", type=float, default=0.20)
    ap.add_argument("--max_val_anom_ratio_quantile", type=float, default=0.80)
    ap.add_argument("--min_pad_floor", type=float, default=1.0)
    ap.add_argument("--max_pool_anom_ratio_cap", type=float, default=0.10)
    ap.add_argument("--min_val_floor", type=int, default=32)
    ap.add_argument("--min_val_anom_floor", type=int, default=7)
    ap.add_argument("--max_val_anom_ratio_cap", type=float, default=0.40)
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
    selected_rows = select_rows(rows, thresholds)
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (
            safe_int(row.get("global_pad_rank"), default=10**9),
            str(row.get("pair_id", "")),
        ),
    )

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
        meta["explainable_midband_rule"] = {
            "thresholds": thresholds,
            "rule_text": build_rule_text(thresholds),
        }
        manifest.append(meta)

    audit = audit_selection(
        selected_rows,
        Path(args.benchmark_summary) if args.benchmark_summary else None,
    )

    summary = {
        "source_rows_json": str(selection_rows_path),
        "protocol_dir": str(protocol_dir),
        "rule_name": "cross_entity_hard_learnable_midband",
        "thresholds": thresholds,
        "rule_text": build_rule_text(thresholds),
        "selected_count": len(selected_rows),
        "missing_meta_count": len(missing),
        "missing_meta": missing,
        "audit": audit,
    }

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "selection_summary.json", summary)
    write_report(out_dir / "REPORT.md", args.title, thresholds, selected_rows, audit)

    print(f"[DONE] Saved manifest: {out_dir / 'manifest.json'}")
    print(f"[DONE] Selected pairs: {len(selected_rows)}")
    print(f"[DONE] Rule: {summary['rule_text']}")


if __name__ == "__main__":
    main()
