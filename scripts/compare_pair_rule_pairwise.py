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


def fmt(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return ""


def build_index(summary: dict):
    rows = summary.get("rows") or []
    return {str(row.get("pair_id")): row for row in rows if row.get("pair_id")}


def get_auroc_delta(row: dict | None, key: str):
    if not row:
        return None
    return float_or_none(((row.get(key) or {}).get("auroc")))


def get_metric(row: dict | None, mode_key: str, metric: str):
    if not row:
        return None
    return float_or_none(((row.get(mode_key) or {}).get(metric)))


def state_from_row(row: dict | None):
    if not row:
        return "not_selected"
    comb = get_auroc_delta(row, "delta_combined_minus_source")
    nas = get_auroc_delta(row, "delta_nas_minus_best_fixed")
    comb_win = comb is not None and comb > 0
    nas_win = nas is not None and nas > 0
    if comb_win and nas_win:
        return "both_win"
    if comb_win:
        return "combined_win_nas_miss"
    if nas_win:
        return "combined_miss_nas_win"
    return "both_miss"


def state_label(state: str):
    return {
        "not_selected": "not selected",
        "both_win": "combined win, NAS win",
        "combined_win_nas_miss": "combined win, NAS miss",
        "combined_miss_nas_win": "combined miss, NAS win",
        "both_miss": "combined miss, NAS miss",
    }.get(state, state)


def issue_label(state: str):
    return {
        "not_selected": "not_selected",
        "both_win": "clean",
        "combined_win_nas_miss": "NAS_miss",
        "combined_miss_nas_win": "combined_miss",
        "both_miss": "combined_miss + NAS_miss",
    }.get(state, state)


def change_tags(old_row: dict | None, new_row: dict | None):
    old_state = state_from_row(old_row)
    new_state = state_from_row(new_row)
    tags = []
    if old_row is None and new_row is not None:
        return [f"new_pair:{new_state}"]
    if old_row is not None and new_row is None:
        return [f"dropped_old_pair:{old_state}"]
    if old_row is None and new_row is None:
        return ["missing_both"]

    old_comb = get_auroc_delta(old_row, "delta_combined_minus_source")
    new_comb = get_auroc_delta(new_row, "delta_combined_minus_source")
    old_nas = get_auroc_delta(old_row, "delta_nas_minus_best_fixed")
    new_nas = get_auroc_delta(new_row, "delta_nas_minus_best_fixed")

    old_comb_win = old_comb is not None and old_comb > 0
    new_comb_win = new_comb is not None and new_comb > 0
    old_nas_win = old_nas is not None and old_nas > 0
    new_nas_win = new_nas is not None and new_nas > 0

    if (not old_comb_win) and new_comb_win:
        tags.append("rescued_combined")
    if old_comb_win and (not new_comb_win):
        tags.append("lost_combined")
    if (not old_nas_win) and new_nas_win:
        tags.append("rescued_nas")
    if old_nas_win and (not new_nas_win):
        tags.append("lost_nas")
    if not tags:
        tags.append("stable")
    return tags


def change_priority(tags):
    joined = ",".join(tags)
    if "rescued_combined" in joined and "rescued_nas" in joined:
        return 0
    if "rescued_combined" in joined or "rescued_nas" in joined:
        return 1
    if "new_pair" in joined:
        return 2
    if "stable" in joined:
        return 3
    if "dropped_old_pair" in joined:
        return 4
    if "lost_" in joined:
        return 5
    return 6


def pair_record(pair_id: str, old_row: dict | None, new_row: dict | None):
    old_state = state_from_row(old_row)
    new_state = state_from_row(new_row)
    return {
        "pair_id": pair_id,
        "old_selected": "Y" if old_row else "",
        "new_selected": "Y" if new_row else "",
        "old_winner": (old_row or {}).get("winner_arch", ""),
        "new_winner": (new_row or {}).get("winner_arch", ""),
        "old_best_nas": (old_row or {}).get("best_nas_arch", ""),
        "new_best_nas": (new_row or {}).get("best_nas_arch", ""),
        "old_d_combined_source": get_auroc_delta(old_row, "delta_combined_minus_source"),
        "new_d_combined_source": get_auroc_delta(new_row, "delta_combined_minus_source"),
        "old_d_nas_fixed": get_auroc_delta(old_row, "delta_nas_minus_best_fixed"),
        "new_d_nas_fixed": get_auroc_delta(new_row, "delta_nas_minus_best_fixed"),
        "old_state": old_state,
        "new_state": new_state,
        "old_issue": issue_label(old_state),
        "new_issue": issue_label(new_state),
        "change_tags": change_tags(old_row, new_row),
        "old_source_auroc": get_metric(old_row, "uad_source", "auroc"),
        "old_combined_auroc": get_metric(old_row, "combined", "auroc"),
        "old_best_nas_auroc": get_metric(old_row, "best_nas", "auroc"),
        "old_best_fixed_auroc": get_metric(old_row, "best_fixed", "auroc"),
        "new_source_auroc": get_metric(new_row, "uad_source", "auroc"),
        "new_combined_auroc": get_metric(new_row, "combined", "auroc"),
        "new_best_nas_auroc": get_metric(new_row, "best_nas", "auroc"),
        "new_best_fixed_auroc": get_metric(new_row, "best_fixed", "auroc"),
    }


def summarize_records(records):
    return {
        "pairs": len(records),
        "old_selected": sum(1 for r in records if r["old_selected"] == "Y"),
        "new_selected": sum(1 for r in records if r["new_selected"] == "Y"),
        "overlap": sum(1 for r in records if r["old_selected"] == "Y" and r["new_selected"] == "Y"),
        "rescued_combined": sum(1 for r in records if "rescued_combined" in r["change_tags"]),
        "rescued_nas": sum(1 for r in records if "rescued_nas" in r["change_tags"]),
        "lost_combined": sum(1 for r in records if "lost_combined" in r["change_tags"]),
        "lost_nas": sum(1 for r in records if "lost_nas" in r["change_tags"]),
        "new_clean": sum(1 for r in records if r["new_state"] == "both_win"),
        "new_combined_miss": sum(1 for r in records if r["new_selected"] == "Y" and r["new_state"] in {"combined_miss_nas_win", "both_miss"}),
        "new_nas_miss": sum(1 for r in records if r["new_selected"] == "Y" and r["new_state"] in {"combined_win_nas_miss", "both_miss"}),
    }


def write_section(lines, dataset_name: str, records, old_label: str, new_label: str):
    stats = summarize_records(records)
    lines.extend([f"## {dataset_name.upper()}", ""])
    lines.append("| Dataset | Total union pairs | Old selected | New selected | Overlap | Rescued combined | Rescued NAS | Lost combined | Lost NAS | New clean (both win) | New combined miss | New NAS miss |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        "| "
        + " | ".join(
            [
                dataset_name,
                str(stats["pairs"]),
                str(stats["old_selected"]),
                str(stats["new_selected"]),
                str(stats["overlap"]),
                str(stats["rescued_combined"]),
                str(stats["rescued_nas"]),
                str(stats["lost_combined"]),
                str(stats["lost_nas"]),
                str(stats["new_clean"]),
                str(stats["new_combined_miss"]),
                str(stats["new_nas_miss"]),
            ]
        )
        + " |"
    )
    lines.extend(["", f"| Pair | {old_label} | {new_label} | Old winner | New winner | Old d_combined-source | New d_combined-source | Old d_NAS-fixed | New d_NAS-fixed | Old state | New state | New issue | Change |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"])

    ordered = sorted(
        records,
        key=lambda r: (
            change_priority(r["change_tags"]),
            -(1 if r["new_selected"] == "Y" else 0),
            -(1 if r["old_selected"] == "Y" else 0),
            r["pair_id"],
        ),
    )
    for r in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["pair_id"],
                    r["old_selected"],
                    r["new_selected"],
                    str(r["old_winner"]),
                    str(r["new_winner"]),
                    fmt(r["old_d_combined_source"]),
                    fmt(r["new_d_combined_source"]),
                    fmt(r["old_d_nas_fixed"]),
                    fmt(r["new_d_nas_fixed"]),
                    state_label(r["old_state"]),
                    state_label(r["new_state"]),
                    str(r["new_issue"]),
                    ", ".join(r["change_tags"]),
                ]
            )
            + " |"
        )


def parse_args():
    ap = argparse.ArgumentParser(description="Compare old vs new pair-rule benchmark summaries pair-by-pair.")
    ap.add_argument("--old_label", default="old_rule")
    ap.add_argument("--new_label", default="adaptation_window")
    ap.add_argument(
        "--section",
        action="append",
        nargs=3,
        metavar=("DATASET", "OLD_SUMMARY", "NEW_SUMMARY"),
        required=True,
        help="Add a dataset section with old and new summary.json paths.",
    )
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    lines = [
        f"# Pairwise Rule Comparison: {args.old_label} vs {args.new_label}",
        "",
        "This table is built on the union of pairs selected by the two rules. ",
        "A pair is marked as rescued when the new rule flips a previously non-positive AUROC delta into a positive one.",
        "",
    ]

    for dataset_name, old_summary_path, new_summary_path in args.section:
        old_summary = read_json(Path(old_summary_path))
        new_summary = read_json(Path(new_summary_path))
        old_idx = build_index(old_summary)
        new_idx = build_index(new_summary)
        pair_ids = sorted(set(old_idx) | set(new_idx))
        records = [pair_record(pair_id, old_idx.get(pair_id), new_idx.get(pair_id)) for pair_id in pair_ids]
        write_section(lines, dataset_name, records, args.old_label, args.new_label)
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[DONE] Wrote pairwise comparison to {out_path}")


if __name__ == "__main__":
    main()
