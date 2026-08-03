import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(
        description="Export all eligible pair split_metadata.json rows from a selection_summary.json into a manifest.json."
    )
    ap.add_argument("--selection_summary", required=True)
    ap.add_argument(
        "--protocol_dir",
        required=True,
        help="Directory containing per-pair split folders, e.g. data/smd_experiments/cross_entity_hard_learnable_qband",
    )
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--require_build_success", action="store_true", default=True)
    args = ap.parse_args()

    selection_summary_path = Path(args.selection_summary)
    protocol_dir = Path(args.protocol_dir)
    out_manifest = Path(args.out_manifest)

    payload = read_json(selection_summary_path)
    rows = payload.get("rows") or []

    manifest = []
    missing = []
    for row in rows:
        if not row.get("eligible"):
            continue
        if args.require_build_success and not row.get("build_success"):
            continue
        pair_id = row.get("pair_id")
        shift_level = row.get("shift_level")
        if not pair_id or not shift_level:
            continue
        meta_path = protocol_dir / f"{pair_id}__{shift_level}" / "split_metadata.json"
        if not meta_path.exists():
            missing.append(str(meta_path))
            continue
        meta = read_json(meta_path)
        meta["selection_export_rule"] = payload.get("rule_name")
        meta["selection_export_source"] = str(selection_summary_path)
        manifest.append(meta)

    manifest.sort(
        key=lambda item: (
            int(item.get("global_pad_rank", 10**9)),
            str(item.get("source_entity", "")),
            str(item.get("target_entity", "")),
        )
    )
    write_json(out_manifest, manifest)

    print(f"[DONE] Saved manifest: {out_manifest}")
    print(f"[DONE] Eligible rows exported: {len(manifest)}")
    if missing:
        print(f"[WARN] Missing split_metadata for {len(missing)} eligible rows.")
        for path in missing[:10]:
            print(f"[WARN] Missing: {path}")


if __name__ == "__main__":
    main()
