import argparse
import json
import math
from pathlib import Path

import numpy as np


RULE_NAME = "cross_entity_val_rich_less_fixed_friendly"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def clamp01(value):
    return max(0.0, min(1.0, safe_float(value, default=0.0)))


def normalize_ratio(value, ref):
    value = safe_float(value, default=0.0)
    ref = safe_float(ref, default=1.0)
    if ref <= 0:
        return 0.0
    return clamp01(value / ref)


def anomaly_segments(y: np.ndarray):
    y = (np.asarray(y).astype(int) > 0).astype(int)
    segments = []
    start = None
    for idx, flag in enumerate(y):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(y) - 1))
    lengths = [end - start + 1 for start, end in segments]
    total_anom = int(y.sum())
    return {
        "segment_count": int(len(segments)),
        "lengths": lengths,
        "max_len": int(max(lengths) if lengths else 0),
        "mean_len": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "total_anom": total_anom,
        "max_len_ratio": float((max(lengths) / total_anom) if lengths and total_anom > 0 else 1.0),
    }


def load_labels(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    if "y" not in data:
        raise ValueError(f"{npz_path} missing y")
    return data["y"]


def compute_event_diversity(val_stats: dict, test_stats: dict, *, event_ref: int):
    val_seg = normalize_ratio(val_stats["segment_count"], event_ref)
    test_seg = normalize_ratio(test_stats["segment_count"], event_ref)
    val_fragmentation = clamp01(1.0 - safe_float(val_stats["max_len_ratio"], default=1.0))
    test_fragmentation = clamp01(1.0 - safe_float(test_stats["max_len_ratio"], default=1.0))
    score = 0.4 * val_seg + 0.2 * test_seg + 0.25 * val_fragmentation + 0.15 * test_fragmentation
    return {
        "score": float(score),
        "components": {
            "val_segment_score": float(val_seg),
            "test_segment_score": float(test_seg),
            "val_fragmentation": float(val_fragmentation),
            "test_fragmentation": float(test_fragmentation),
        },
        "raw": {
            "val": val_stats,
            "test": test_stats,
            "event_ref": int(event_ref),
        },
    }


def md_table(path: Path, rows, columns, title: str):
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float) and math.isfinite(value):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value) if value is not None else "")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Rerank val-rich candidates toward less fixed-friendly pairs.")
    ap.add_argument("--protocol_dir", required=True, help="Existing protocol dir from cross_entity_learnable_shift_val_rich")
    ap.add_argument("--out_dir", required=True, help="Output protocol dir for reranked manifest")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--target_repeat_penalty", type=float, default=0.08)
    ap.add_argument("--event_ref", type=int, default=4)
    ap.add_argument("--min_val_segments", type=int, default=2)
    args = ap.parse_args()

    protocol_dir = Path(args.protocol_dir)
    out_dir = Path(args.out_dir)
    summary_path = protocol_dir / "selection_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    selection_summary = read_json(summary_path)
    rows = selection_summary.get("rows") or []
    row_index = {}
    for row in rows:
        row_index[(row.get("pair_id"), row.get("shift_level"))] = row
    candidates = []
    for row in rows:
        if not row.get("build_success") or not row.get("eligible"):
            continue
        pair_id = row["pair_id"]
        shift_level = row["shift_level"]
        split_dir = protocol_dir / f"{pair_id}__{shift_level}"
        meta_path = split_dir / "split_metadata.json"
        val_path = split_dir / "val_mixed.npz"
        test_path = split_dir / "test_mixed.npz"
        if not meta_path.exists() or not val_path.exists() or not test_path.exists():
            continue
        meta = read_json(meta_path)
        source_row = row_index.get((pair_id, shift_level), {})
        val_stats = anomaly_segments(load_labels(val_path))
        test_stats = anomaly_segments(load_labels(test_path))
        if val_stats["segment_count"] < args.min_val_segments:
            continue
        base = meta.get("learnable_shift_val_rich") or {}
        base_components = base.get("components") or {}
        event_info = compute_event_diversity(val_stats, test_stats, event_ref=args.event_ref)
        score = (
            0.30 * safe_float(base_components.get("shift_strength"))
            + 0.20 * safe_float(base_components.get("val_richness"))
            + 0.15 * safe_float(base_components.get("pool_cleanliness"))
            + 0.35 * safe_float(event_info.get("score"))
        )
        target_entity = meta.get("target_entity")
        source_entity = meta.get("source_entity")
        candidates.append(
            {
                "pair_id": pair_id,
                "source_entity": source_entity,
                "target_entity": target_entity,
                "shift_level": shift_level,
                "base_score": safe_float(base.get("score")),
                "refined_score": float(score),
                "event_diversity_score": safe_float(event_info.get("score")),
                "val_segment_count": val_stats["segment_count"],
                "test_segment_count": test_stats["segment_count"],
                "val_max_len_ratio": safe_float(val_stats["max_len_ratio"]),
                "test_max_len_ratio": safe_float(test_stats["max_len_ratio"]),
                "global_pad_rank": source_row.get("global_pad_rank"),
                "pad_value": source_row.get("pad_value"),
                "meta": meta,
                "event_info": event_info,
            }
        )

    candidates.sort(
        key=lambda row: (
            row["refined_score"],
            row["event_diversity_score"],
            -safe_float(row.get("global_pad_rank"), default=10**9),
        ),
        reverse=True,
    )

    selected = []
    used_targets = {}
    remaining = candidates[:]
    while remaining and len(selected) < args.topk:
        best_idx = None
        best_score = None
        for idx, row in enumerate(remaining):
            penalty = args.target_repeat_penalty * used_targets.get(row["target_entity"], 0)
            greedy_score = row["refined_score"] - penalty
            if best_score is None or greedy_score > best_score:
                best_score = greedy_score
                best_idx = idx
        chosen = remaining.pop(best_idx)
        chosen["greedy_score"] = float(best_score)
        selected.append(chosen)
        used_targets[chosen["target_entity"]] = used_targets.get(chosen["target_entity"], 0) + 1

    manifest = []
    for selection_order, row in enumerate(selected, start=1):
        meta = dict(row["meta"])
        meta["ranking_rule"] = RULE_NAME
        meta["selection_order"] = selection_order
        meta["selected_shift_level"] = row["shift_level"]
        meta["global_pad_rank"] = row["global_pad_rank"]
        meta["candidate_pair_shift_precheck"] = {"pad_value": row["pad_value"]}
        meta["less_fixed_friendly"] = {
            "refined_score": row["refined_score"],
            "greedy_score": row["greedy_score"],
            "event_diversity": row["event_info"],
            "target_repeat_penalty": args.target_repeat_penalty,
        }
        manifest.append(meta)

    serializable_rows = []
    selected_names = {row["pair_id"] for row in selected}
    selected_shift = {(row["pair_id"], row["shift_level"]) for row in selected}
    for row in candidates:
        serializable_rows.append(
            {
                "pair_id": row["pair_id"],
                "source_entity": row["source_entity"],
                "target_entity": row["target_entity"],
                "shift_level": row["shift_level"],
                "global_pad_rank": row["global_pad_rank"],
                "pad_value": row["pad_value"],
                "base_score": row["base_score"],
                "refined_score": row["refined_score"],
                "event_diversity_score": row["event_diversity_score"],
                "val_segment_count": row["val_segment_count"],
                "test_segment_count": row["test_segment_count"],
                "val_max_len_ratio": row["val_max_len_ratio"],
                "test_max_len_ratio": row["test_max_len_ratio"],
                "selected_pair": row["pair_id"] in selected_names,
                "selected_split": (row["pair_id"], row["shift_level"]) in selected_shift,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "selection_summary.json",
        {
            "rule_name": RULE_NAME,
            "source_protocol_dir": str(protocol_dir),
            "topk": args.topk,
            "target_repeat_penalty": args.target_repeat_penalty,
            "event_ref": args.event_ref,
            "min_val_segments": args.min_val_segments,
            "rows": serializable_rows,
        },
    )
    md_table(
        out_dir / "selection_summary.md",
        serializable_rows,
        [
            "pair_id",
            "shift_level",
            "global_pad_rank",
            "pad_value",
            "base_score",
            "refined_score",
            "event_diversity_score",
            "val_segment_count",
            "test_segment_count",
            "val_max_len_ratio",
            "test_max_len_ratio",
            "selected_pair",
            "selected_split",
        ],
        title="Val-Rich Less Fixed-Friendly Selection",
    )
    write_json(out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
