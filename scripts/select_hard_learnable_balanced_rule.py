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
    RULE_CROSS_ENTITY_HARD_LEARNABLE_BALANCED,
    public_rows,
    safe_float,
    safe_int,
    save_json,
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def quantile(values, q):
    finite = sorted(
        safe_float(value)
        for value in values
        if math.isfinite(safe_float(value))
    )
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return float(finite[0])
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    if lo == hi:
        return float(finite[lo])
    return float(finite[lo] * (1.0 - frac) + finite[hi] * frac)


def normalize_linear(value, low, high):
    value = safe_float(value, default=float("nan"))
    low = safe_float(low, default=float("nan"))
    high = safe_float(high, default=float("nan"))
    if not math.isfinite(value) or not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return 0.0
    return max(0.0, min(1.0, float((value - low) / (high - low))))


def load_rows(path: Path):
    payload = read_json(path)
    rows = payload.get("rows") or []
    return payload, rows


def find_meta_path(protocol_dir: Path, pair_id: str, shift_level: str):
    direct = protocol_dir / f"{pair_id}__{shift_level}" / "split_metadata.json"
    if direct.exists():
        return direct
    nested = list(protocol_dir.glob(f"**/{pair_id}__{shift_level}/split_metadata.json"))
    if nested:
        return nested[0]
    return None


def compute_thresholds(rows, args):
    buildable = [row for row in rows if row.get("build_success")]
    thresholds = {
        "min_pad_value": args.min_pad_value if args.min_pad_value is not None else float("nan"),
        "max_pad_value": args.max_pad_value if args.max_pad_value is not None else float("nan"),
        "max_pool_anom_ratio": args.max_pool_anom_ratio if args.max_pool_anom_ratio is not None else float("nan"),
        "min_val_count": args.min_val_count if args.min_val_count is not None else 0,
        "min_test_count": args.min_test_count if args.min_test_count is not None else 0,
        "min_val_anomaly_count": args.min_val_anomaly_count if args.min_val_anomaly_count is not None else 0,
        "max_precheck_l2": args.max_precheck_l2 if args.max_precheck_l2 is not None else float("nan"),
    }
    if math.isnan(safe_float(thresholds["min_pad_value"])):
        thresholds["min_pad_value"] = quantile([row.get("pad_value") for row in buildable], args.min_pad_quantile)
    if math.isnan(safe_float(thresholds["max_pad_value"])) and args.max_pad_quantile is not None:
        thresholds["max_pad_value"] = quantile([row.get("pad_value") for row in buildable], args.max_pad_quantile)
    if math.isnan(safe_float(thresholds["max_pool_anom_ratio"])):
        thresholds["max_pool_anom_ratio"] = min(
            args.max_pool_anom_ratio_cap,
            quantile([row.get("target_pool_hidden_anomaly_ratio") for row in buildable], args.max_pool_anom_ratio_quantile),
        )
    if not thresholds["min_val_count"]:
        thresholds["min_val_count"] = max(
            args.min_val_count_floor,
            int(math.ceil(quantile([row.get("val_count") for row in buildable], args.min_val_count_quantile))),
        )
    if not thresholds["min_val_anomaly_count"]:
        thresholds["min_val_anomaly_count"] = max(
            args.min_val_anomaly_floor,
            int(math.ceil(quantile([row.get("val_anomaly_count") for row in buildable], args.min_val_anomaly_quantile))),
        )
    if math.isnan(safe_float(thresholds["max_precheck_l2"])):
        thresholds["max_precheck_l2"] = quantile([row.get("pad_feature_mean_l2") for row in buildable], args.max_precheck_l2_quantile)

    val_ratio_values = []
    for row in buildable:
        val_count = safe_int(row.get("val_count"), default=0)
        val_anom = safe_int(row.get("val_anomaly_count"), default=0)
        if val_count > 0:
            val_ratio_values.append(float(val_anom / val_count))
    min_val_ratio = quantile(val_ratio_values, args.min_val_anomaly_ratio_quantile)
    max_val_ratio = quantile(val_ratio_values, args.max_val_anomaly_ratio_quantile)
    thresholds["min_val_anomaly_ratio"] = max(args.min_val_anomaly_ratio_floor, safe_float(min_val_ratio, default=args.min_val_anomaly_ratio_floor))
    thresholds["max_val_anomaly_ratio"] = min(args.max_val_anomaly_ratio_cap, safe_float(max_val_ratio, default=args.max_val_anomaly_ratio_cap))
    if thresholds["min_val_anomaly_ratio"] > thresholds["max_val_anomaly_ratio"]:
        thresholds["min_val_anomaly_ratio"] = args.min_val_anomaly_ratio_floor
        thresholds["max_val_anomaly_ratio"] = args.max_val_anomaly_ratio_cap
    return thresholds


def enrich_rows(rows):
    enriched = []
    for row in rows:
        item = dict(row)
        val_count = safe_int(item.get("val_count"), default=0)
        val_anom = safe_int(item.get("val_anomaly_count"), default=0)
        item["val_anomaly_ratio"] = float(val_anom / val_count) if val_count > 0 else float("nan")
        enriched.append(item)
    return enriched


def passes_thresholds(row, thresholds):
    if not row.get("build_success"):
        return False
    if safe_float(row.get("pad_value"), default=float("-inf")) < safe_float(thresholds["min_pad_value"], default=float("inf")):
        return False
    max_pad_value = safe_float(thresholds.get("max_pad_value"), default=float("nan"))
    if math.isfinite(max_pad_value) and safe_float(row.get("pad_value"), default=float("inf")) > max_pad_value:
        return False
    if safe_float(row.get("target_pool_hidden_anomaly_ratio"), default=float("inf")) > safe_float(thresholds["max_pool_anom_ratio"], default=float("-inf")):
        return False
    if safe_int(row.get("val_count"), default=0) < safe_int(thresholds["min_val_count"], default=0):
        return False
    if safe_int(row.get("test_count"), default=0) < safe_int(thresholds["min_test_count"], default=0):
        return False
    if safe_int(row.get("val_anomaly_count"), default=0) < safe_int(thresholds["min_val_anomaly_count"], default=0):
        return False
    val_ratio = safe_float(row.get("val_anomaly_ratio"), default=float("nan"))
    if not math.isfinite(val_ratio):
        return False
    if val_ratio < safe_float(thresholds["min_val_anomaly_ratio"], default=0.0):
        return False
    if val_ratio > safe_float(thresholds["max_val_anomaly_ratio"], default=1.0):
        return False
    max_precheck_l2 = safe_float(thresholds["max_precheck_l2"], default=float("nan"))
    if math.isfinite(max_precheck_l2):
        precheck_l2 = safe_float(row.get("pad_feature_mean_l2"), default=float("inf"))
        if not math.isfinite(precheck_l2) or precheck_l2 > max_precheck_l2:
            return False
    return True


def compute_scores(rows, thresholds, args):
    filtered = [row for row in rows if row.get("passes_balanced")]
    if not filtered:
        return
    precheck_vals = [safe_float(row.get("pad_feature_mean_l2"), default=float("nan")) for row in filtered]
    precheck_vals = [val for val in precheck_vals if math.isfinite(val)]
    low_precheck = min(precheck_vals) if precheck_vals else 0.0
    high_precheck = max(precheck_vals) if precheck_vals else 1.0

    for row in filtered:
        base_score = safe_float(row.get("hard_learnable_score"), default=0.0)
        val_ratio = safe_float(row.get("val_anomaly_ratio"), default=1.0)
        precheck_pen = normalize_linear(row.get("pad_feature_mean_l2"), low_precheck, high_precheck)
        row["balanced_score"] = float(
            base_score
            - args.val_ratio_penalty * val_ratio
            - args.precheck_penalty * precheck_pen
        )
        row["precheck_penalty_norm"] = float(precheck_pen)


def build_rule_text(thresholds, args):
    clauses = [f"PAD >= {safe_float(thresholds['min_pad_value']):.4f}"]
    max_pad_value = safe_float(thresholds.get("max_pad_value"), default=float("nan"))
    if math.isfinite(max_pad_value):
        clauses.append(f"PAD <= {max_pad_value:.4f}")
    clauses.extend(
        [
            f"target_pool_hidden_anomaly_ratio <= {safe_float(thresholds['max_pool_anom_ratio']):.4f}",
            f"val_count >= {safe_int(thresholds['min_val_count'])}",
        ]
    )
    min_test_count = safe_int(thresholds.get("min_test_count"), default=0)
    if min_test_count > 0:
        clauses.append(f"test_count >= {min_test_count}")
    clauses.extend(
        [
            f"val_anomaly_count >= {safe_int(thresholds['min_val_anomaly_count'])}",
            (
                f"{safe_float(thresholds['min_val_anomaly_ratio']):.4f} <= "
                f"val_anomaly_ratio <= {safe_float(thresholds['max_val_anomaly_ratio']):.4f}"
            ),
        ]
    )
    max_precheck_l2 = safe_float(thresholds["max_precheck_l2"], default=float("nan"))
    if math.isfinite(max_precheck_l2):
        clauses.append(f"precheck feature-mean L2 <= {max_precheck_l2:.4f}")
    return (
        "Starting from buildable cross-entity hard pairs, we retained only pairs with "
        + ", ".join(clauses[:-1])
        + f", and {clauses[-1]}. "
        + f"Pairs were then ranked by balanced_score = hard_learnable_score - {args.val_ratio_penalty:.2f}*val_anomaly_ratio "
        + f"- {args.precheck_penalty:.2f}*normalized_precheck_l2, with optional per-target and per-source caps to avoid near-duplicate tasks."
    )


def select_rows(rows, args):
    selected = []
    target_counts = {}
    source_counts = {}
    ranked = sorted(
        [row for row in rows if row.get("passes_balanced")],
        key=lambda row: (
            safe_float(row.get("balanced_score"), default=float("-inf")),
            safe_float(row.get("pool_cleanliness"), default=float("-inf")),
            safe_float(row.get("val_richness"), default=float("-inf")),
            -safe_float(row.get("pad_feature_mean_l2"), default=float("inf")),
            -safe_int(row.get("global_pad_rank"), default=10**9),
        ),
        reverse=True,
    )
    for row in ranked:
        source_entity = str(row.get("source_entity", ""))
        target_entity = str(row.get("target_entity", ""))
        if args.max_per_target > 0 and target_counts.get(target_entity, 0) >= args.max_per_target:
            continue
        if args.max_per_source > 0 and source_counts.get(source_entity, 0) >= args.max_per_source:
            continue
        selected.append(row)
        target_counts[target_entity] = target_counts.get(target_entity, 0) + 1
        source_counts[source_entity] = source_counts.get(source_entity, 0) + 1
        if args.max_selected > 0 and len(selected) >= args.max_selected:
            break
    return selected


def write_report(path: Path, title: str, thresholds: dict, selected_rows, all_rows, args):
    lines = [
        f"# {title}",
        "",
        "## Rule",
        "",
        f"`{build_rule_text(thresholds, args)}`",
        "",
        f"- Buildable rows: {sum(1 for row in all_rows if row.get('build_success'))}",
        f"- Eligible balanced rows: {sum(1 for row in all_rows if row.get('passes_balanced'))}",
        f"- Selected rows: {len(selected_rows)}",
        "",
        "## Selected Pairs",
        "",
        "| Order | PAD rank | Pair | Shift | PAD | balanced_score | hard_learnable_score | val_ratio | pool ratio | val count | val anom | precheck_l2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(selected_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("global_pad_rank", "")),
                    str(row.get("pair_id", "")),
                    str(row.get("shift_level", "")),
                    f"{safe_float(row.get('pad_value')):.4f}",
                    f"{safe_float(row.get('balanced_score')):.4f}",
                    f"{safe_float(row.get('hard_learnable_score')):.4f}",
                    f"{safe_float(row.get('val_anomaly_ratio')):.4f}",
                    f"{safe_float(row.get('target_pool_hidden_anomaly_ratio')):.4f}",
                    str(safe_int(row.get("val_count"))),
                    str(safe_int(row.get("val_anomaly_count"))),
                    f"{safe_float(row.get('pad_feature_mean_l2')):.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Refine the hard-learnable pair pool into a more balanced benchmark manifest.")
    ap.add_argument("--selection_summary_json", required=True)
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Cross-Entity Hard Learnable Balanced Selection")
    ap.add_argument("--rule_name", default=RULE_CROSS_ENTITY_HARD_LEARNABLE_BALANCED)
    ap.add_argument("--meta_key", default="hard_learnable_balanced")
    ap.add_argument("--min_pad_value", type=float, default=None)
    ap.add_argument("--min_pad_quantile", type=float, default=0.0)
    ap.add_argument("--max_pad_value", type=float, default=None)
    ap.add_argument("--max_pad_quantile", type=float, default=None)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=None)
    ap.add_argument("--max_pool_anom_ratio_quantile", type=float, default=1.0)
    ap.add_argument("--max_pool_anom_ratio_cap", type=float, default=0.10)
    ap.add_argument("--min_val_count", type=int, default=None)
    ap.add_argument("--min_val_count_quantile", type=float, default=0.0)
    ap.add_argument("--min_val_count_floor", type=int, default=32)
    ap.add_argument("--min_test_count", type=int, default=None)
    ap.add_argument("--min_val_anomaly_count", type=int, default=None)
    ap.add_argument("--min_val_anomaly_quantile", type=float, default=0.0)
    ap.add_argument("--min_val_anomaly_floor", type=int, default=7)
    ap.add_argument("--max_precheck_l2", type=float, default=None)
    ap.add_argument("--max_precheck_l2_quantile", type=float, default=1.0)
    ap.add_argument("--min_val_anomaly_ratio_quantile", type=float, default=0.20)
    ap.add_argument("--min_val_anomaly_ratio_floor", type=float, default=0.0)
    ap.add_argument("--max_val_anomaly_ratio_quantile", type=float, default=0.90)
    ap.add_argument("--max_val_anomaly_ratio_cap", type=float, default=0.40)
    ap.add_argument("--val_ratio_penalty", type=float, default=0.40)
    ap.add_argument("--precheck_penalty", type=float, default=0.10)
    ap.add_argument("--max_per_target", type=int, default=0)
    ap.add_argument("--max_per_source", type=int, default=0)
    ap.add_argument("--max_selected", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.selection_summary_json)
    protocol_dir = Path(args.protocol_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    payload, rows = load_rows(summary_path)
    rows = enrich_rows(rows)
    thresholds = compute_thresholds(rows, args)

    for row in rows:
        row["passes_balanced"] = passes_thresholds(row, thresholds)
        row["balanced_score"] = None
        row["precheck_penalty_norm"] = None
        row["selected"] = False
    compute_scores(rows, thresholds, args)
    selected_rows = select_rows(rows, args)
    for row in selected_rows:
        row["selected"] = True

    manifest = []
    missing_meta = []
    for selection_order, row in enumerate(selected_rows, start=1):
        meta_path = find_meta_path(protocol_dir, row["pair_id"], row["shift_level"])
        if meta_path is None:
            missing_meta.append(f"{row['pair_id']}::{row['shift_level']}")
            continue
        meta = read_json(meta_path)
        meta["selection_order"] = selection_order
        meta["selected_shift_level"] = row["shift_level"]
        meta[args.meta_key] = {
            "rule_name": args.rule_name,
            "balanced_score": row["balanced_score"],
            "hard_learnable_score": row.get("hard_learnable_score"),
            "val_anomaly_ratio": row["val_anomaly_ratio"],
            "precheck_l2": row.get("pad_feature_mean_l2"),
            "thresholds": thresholds,
            "rule_text": build_rule_text(thresholds, args),
        }
        manifest.append(meta)

    summary = {
        "source_selection_summary": str(summary_path),
        "protocol_dir": str(protocol_dir),
        "rule_name": args.rule_name,
        "rule_text": build_rule_text(thresholds, args),
        "thresholds": thresholds,
        "config": {
            "val_ratio_penalty": args.val_ratio_penalty,
            "precheck_penalty": args.precheck_penalty,
            "max_per_target": args.max_per_target,
            "max_per_source": args.max_per_source,
            "max_selected": args.max_selected,
            "meta_key": args.meta_key,
        },
        "selected_count": len(selected_rows),
        "missing_meta_count": len(missing_meta),
        "missing_meta": missing_meta,
        "rows": public_rows(rows),
    }

    save_json(out_dir / "manifest.json", manifest)
    save_json(out_dir / "selection_summary.json", summary)
    write_report(out_dir / "REPORT.md", args.title, thresholds, selected_rows, rows, args)

    print(f"[DONE] Saved manifest: {out_dir / 'manifest.json'}")
    print(f"[DONE] Selected pairs: {len(selected_rows)}")
    print(f"[DONE] Rule: {build_rule_text(thresholds, args)}")


if __name__ == "__main__":
    main()
