import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.build_pair_rule_common import (
    RULE_CROSS_ENTITY_HARD_LEARNABLE,
    RULE_CROSS_ENTITY_HARD_LEARNABLE_QBAND,
    RULE_CROSS_ENTITY_LEARNABLE_SHIFT_VAL_RICH,
    RULE_CROSS_ENTITY_PAPER_SAFE,
    build_hard_learnable_methods_text,
    build_hard_learnable_qband_methods_text,
    build_paper_safe_methods_text,
    compute_hard_learnable_score,
    compute_hard_learnable_qband_thresholds,
    compute_learnable_shift_val_rich_score,
    compute_paper_safe_thresholds,
    finite_quantile,
    load_pad_pair_rankings,
    public_rows,
    save_json,
    safe_float,
    safe_int,
    safe_ratio,
    write_table_csv,
    write_table_markdown,
)
from scripts.make_uad_smd import (
    DEFAULT_GUARD,
    DEFAULT_MAX_POOL_ANOM_RATIO,
    DEFAULT_MIN_ANOM_TEST,
    DEFAULT_MIN_ANOM_VAL,
    DEFAULT_MIN_TARGET_POOL,
    DEFAULT_MIN_TEST,
    DEFAULT_MIN_VAL,
    DEFAULT_SEARCH_STEP,
    DEFAULT_TARGET_POOL_FRAC,
    DEFAULT_VAL_FRAC,
    binarize_y,
    compute_domain_shift_metrics,
    create_dataset,
    load_npz,
)


def list_entity_dirs(data_root: Path):
    return sorted([p for p in data_root.iterdir() if p.is_dir() and (p / "source.npz").exists() and (p / "target.npz").exists()])


def read_entity_meta(entity_dir: Path) -> dict:
    meta_path = entity_dir / "metadata.json"
    if not meta_path.exists():
        name = entity_dir.name
        role = "train" if "train" in name.lower() else "test"
        version_tag = name.split("-")[0] if "-" in name else name
        return {"entity_name": name, "split_role": role, "version_tag": version_tag}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def source_norm_windows(entity_dir: Path):
    Xs, ys = load_npz(str(entity_dir / "source.npz"))
    ys = binarize_y(ys)
    return Xs[ys == 0]


def target_windows(entity_dir: Path):
    Xt, yt = load_npz(str(entity_dir / "target.npz"))
    yt = binarize_y(yt)
    return Xt, yt


def candidate_targets(source_dir: Path, all_dirs, same_version_only: bool):
    src_meta = read_entity_meta(source_dir)
    src_version = src_meta.get("version_tag") or source_dir.name.split("-")[0]
    out = []
    for target_dir in all_dirs:
        if target_dir == source_dir:
            continue
        target_meta = read_entity_meta(target_dir)
        if str(target_meta.get("split_role", "")).lower() != "test":
            continue
        tgt_version = target_meta.get("version_tag") or target_dir.name.split("-")[0]
        if same_version_only and tgt_version != src_version:
            continue
        out.append(target_dir)
    return out


def rank_cross_targets(source_dir: Path, target_dirs, min_target_anom: int):
    Xs_norm = source_norm_windows(source_dir)
    ranked = []
    for target_dir in target_dirs:
        Xt, yt = target_windows(target_dir)
        target_anom = int((yt == 1).sum())
        if target_anom < min_target_anom:
            continue
        shift = compute_domain_shift_metrics(Xs_norm, Xt, seed=42)
        auc = shift["domain_auc"]
        if not np.isfinite(auc):
            auc = 0.5
        ranked.append((float(auc), target_anom, target_dir, shift))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def build_args(
    *,
    source_dir: Path,
    target_dir: Path,
    out_dir: Path,
    shift_level: str,
    target_pool_frac: float,
    val_frac: float,
    guard: int,
    search_step: int,
    max_pool_anom_ratio: float,
    min_target_pool: int,
    min_val: int,
    min_test: int,
    min_anom_val: int,
    min_anom_test: int,
    seed: int,
):
    return SimpleNamespace(
        machine_dir=str(source_dir),
        target_machine_dir=str(target_dir),
        source_name="source.npz",
        target_name="target.npz",
        out_dir=str(out_dir),
        out_train="train_normal.npz",
        out_target_pool="target_pool_unlabeled.npz",
        out_val="val_mixed.npz",
        out_test="test_mixed.npz",
        out_meta="split_metadata.json",
        split_mode="search",
        shift_level=shift_level,
        train_normal_frac=1.0,
        target_pool_frac=target_pool_frac,
        val_frac=val_frac,
        guard=guard,
        search_step=search_step,
        max_pool_anom_ratio=max_pool_anom_ratio,
        min_train=0,
        min_target_pool=min_target_pool,
        min_val=min_val,
        min_test=min_test,
        min_anom_val=min_anom_val,
        min_anom_test=min_anom_test,
        allow_single_class_eval=False,
        seed=seed,
        strict=False,
    )


def build_hard_learnable_pairs(args, entity_dirs, out_root: Path):
    if not args.rankings_json:
        raise ValueError("--rankings_json is required when --pair_rule is cross_entity_hard_learnable")

    rankings_path = Path(args.rankings_json)
    if not rankings_path.exists():
        raise FileNotFoundError(rankings_path)

    entity_map = {entity_dir.name: entity_dir for entity_dir in entity_dirs}
    entity_meta = {entity_dir.name: read_entity_meta(entity_dir) for entity_dir in entity_dirs}
    protocol_dir = out_root / RULE_CROSS_ENTITY_HARD_LEARNABLE
    protocol_dir.mkdir(parents=True, exist_ok=True)

    ranking_payload, ranked_rows = load_pad_pair_rankings(rankings_path)
    shift_levels = [s.strip() for s in args.hard_rule_shift_levels.split(",") if s.strip()]
    if not shift_levels:
        raise ValueError("--hard_rule_shift_levels must contain at least one shift level")
    if any(level != "hard" for level in shift_levels):
        raise ValueError("--hard_rule_shift_levels must be 'hard' for cross_entity_hard_learnable")

    precheck_l2_cap = finite_quantile(
        [row.get("pad_feature_mean_l2") for row in ranked_rows],
        args.hard_rule_max_precheck_l2_quantile,
    )
    methods_text = build_hard_learnable_methods_text(
        min_val_count=args.hard_rule_min_val,
        min_val_anom=args.hard_rule_min_anom_val,
        max_pool_anom_ratio=args.hard_rule_max_pool_anom_ratio,
        precheck_l2_quantile=args.hard_rule_max_precheck_l2_quantile,
    )

    pair_best = {}
    build_rows = []

    for ranking_row in ranked_rows:
        src_name = ranking_row["source_entity"]
        tgt_name = ranking_row["target_entity"]
        src_meta = entity_meta.get(src_name, {})
        tgt_meta = entity_meta.get(tgt_name, {})
        src_role = str(src_meta.get("split_role", "")).lower()
        tgt_role = str(tgt_meta.get("split_role", "")).lower()
        src_version = src_meta.get("version_tag") or src_name.split("-")[0]
        tgt_version = tgt_meta.get("version_tag") or tgt_name.split("-")[0]
        pair_name = ranking_row["pair_id"]
        precheck_l2 = safe_float(ranking_row.get("pad_feature_mean_l2"), default=float("nan"))
        base_row = {
            "global_pad_rank": ranking_row["global_pad_rank"],
            "source_entity": src_name,
            "target_entity": tgt_name,
            "pair_id": pair_name,
            "pad_value": ranking_row["pad_value"],
            "pad_domain_acc": ranking_row["pad_domain_acc"],
            "pad_domain_auc": ranking_row["pad_domain_auc"],
            "pad_feature_mean_l2": precheck_l2,
            "precheck_l2_cap": precheck_l2_cap,
            "source_role": src_role,
            "target_role": tgt_role,
            "source_version": src_version,
            "target_version": tgt_version,
        }

        failure_reason = ""
        if src_name == tgt_name:
            failure_reason = "same_entity"
        elif src_name not in entity_map or tgt_name not in entity_map:
            failure_reason = "filtered_out_by_entity_subset"
        elif src_role != "train":
            failure_reason = "source_must_be_train"
        elif tgt_role != "test":
            failure_reason = "target_must_be_test"
        elif (not args.allow_cross_version) and tgt_version != src_version:
            failure_reason = "cross_version_blocked"

        if failure_reason:
            build_rows.append(
                {
                    **base_row,
                    "shift_level": "",
                    "build_success": False,
                    "eligible": False,
                    "selected": False,
                    "hard_learnable_score": "",
                    "pad_strength": "",
                    "pool_cleanliness": "",
                    "val_richness": "",
                    "target_pool_hidden_anomaly_ratio": "",
                    "train_normal_count": "",
                    "target_pool_count": "",
                    "val_count": "",
                    "test_count": "",
                    "val_anomaly_count": "",
                    "test_anomaly_count": "",
                    "failure_reason": failure_reason,
                }
            )
            continue

        for shift_level in shift_levels:
            source_dir = entity_map[src_name]
            target_dir = entity_map[tgt_name]
            out_dir = protocol_dir / f"{pair_name}__{shift_level}"
            ds_args = build_args(
                source_dir=source_dir,
                target_dir=target_dir,
                out_dir=out_dir,
                shift_level=shift_level,
                target_pool_frac=args.hard_rule_target_pool_frac,
                val_frac=args.hard_rule_val_frac,
                guard=args.hard_rule_guard,
                search_step=args.hard_rule_search_step,
                max_pool_anom_ratio=args.hard_rule_max_pool_anom_ratio,
                min_target_pool=args.hard_rule_min_target_pool,
                min_val=args.hard_rule_min_val,
                min_test=args.hard_rule_min_test,
                min_anom_val=args.hard_rule_min_anom_val,
                min_anom_test=args.hard_rule_min_anom_test,
                seed=args.seed,
            )
            try:
                meta = create_dataset(ds_args)
                score_info = compute_hard_learnable_score(
                    pad_value=ranking_row["pad_value"],
                    target_pool_hidden_anomaly_ratio=meta.get("target_pool_hidden_anomaly_ratio"),
                    val_count=meta.get("val_count"),
                    val_anomaly_count=meta.get("val_anomaly_count"),
                    max_pool_anom_ratio=args.hard_rule_max_pool_anom_ratio,
                    val_count_ref=args.hard_rule_val_count_ref,
                    val_anom_ref=args.hard_rule_val_anom_ref,
                    min_pad_value=args.hard_rule_min_pad_value,
                    max_pad_value=args.hard_rule_max_pad_value,
                )
                pad_value = safe_float(ranking_row["pad_value"], default=0.0)
                pool_anom_ratio = safe_float(meta.get("target_pool_hidden_anomaly_ratio"), default=1.0)
                val_count = int(meta.get("val_count", 0))
                val_anom_count = int(meta.get("val_anomaly_count", 0))
                precheck_ok = True
                if np.isfinite(precheck_l2_cap):
                    precheck_ok = np.isfinite(precheck_l2) and precheck_l2 <= precheck_l2_cap
                eligible = (
                    pad_value >= args.hard_rule_min_pad_value
                    and pool_anom_ratio <= args.hard_rule_max_pool_anom_ratio
                    and val_count >= args.hard_rule_min_val
                    and val_anom_count >= args.hard_rule_min_anom_val
                    and precheck_ok
                )
                meta["candidate_pair_shift_precheck"] = ranking_row["raw_row"].get("pad_latent", {})
                meta["ranking_rule"] = RULE_CROSS_ENTITY_HARD_LEARNABLE
                meta["ranking_source"] = str(rankings_path)
                meta["global_pad_rank"] = int(ranking_row["global_pad_rank"])
                meta["hard_learnable"] = {
                    **score_info,
                    "precheck_feature_mean_l2": precheck_l2,
                    "precheck_feature_mean_l2_cap": precheck_l2_cap,
                    "methods_text": methods_text,
                }
                row = {
                    **base_row,
                    "shift_level": shift_level,
                    "build_success": True,
                    "eligible": bool(eligible),
                    "selected": False,
                    "hard_learnable_score": score_info["score"],
                    "pad_strength": score_info["components"]["pad_strength"],
                    "pool_cleanliness": score_info["components"]["pool_cleanliness"],
                    "val_richness": score_info["components"]["val_richness"],
                    "target_pool_hidden_anomaly_ratio": meta["target_pool_hidden_anomaly_ratio"],
                    "train_normal_count": meta["train_normal_count"],
                    "target_pool_count": meta["target_pool_count"],
                    "val_count": meta["val_count"],
                    "test_count": meta["test_count"],
                    "val_anomaly_count": meta["val_anomaly_count"],
                    "test_anomaly_count": meta["test_anomaly_count"],
                    "failure_reason": "",
                }
                build_rows.append(row)
                if eligible:
                    current = pair_best.get(pair_name)
                    sort_key = (
                        -int(ranking_row["global_pad_rank"]),
                        int(meta["val_anomaly_count"]),
                        int(meta["val_count"]),
                        float(score_info["components"]["pool_cleanliness"]),
                    )
                    if current is None or sort_key > current["sort_key"]:
                        pair_best[pair_name] = {
                            "sort_key": sort_key,
                            "row": row,
                            "meta": meta,
                        }
            except Exception as exc:
                build_rows.append(
                    {
                        **base_row,
                        "shift_level": shift_level,
                        "build_success": False,
                        "eligible": False,
                        "selected": False,
                        "hard_learnable_score": "",
                        "pad_strength": "",
                        "pool_cleanliness": "",
                        "val_richness": "",
                        "target_pool_hidden_anomaly_ratio": "",
                        "train_normal_count": "",
                        "target_pool_count": "",
                        "val_count": "",
                        "test_count": "",
                        "val_anomaly_count": "",
                        "test_anomaly_count": "",
                        "failure_reason": str(exc),
                    }
                )

    selected_entries = sorted(
        pair_best.values(),
        key=lambda item: item["sort_key"],
        reverse=True,
    )[: args.global_topk]

    manifest = []
    selected_pair_names = {entry["row"]["pair_id"] for entry in selected_entries}
    for selection_order, entry in enumerate(selected_entries, start=1):
        entry["row"]["selected"] = True
        meta = entry["meta"]
        meta["selection_order"] = selection_order
        meta["selected_shift_level"] = entry["row"]["shift_level"]
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "shift_level",
        "source_role",
        "target_role",
        "source_version",
        "target_version",
        "pad_value",
        "pad_feature_mean_l2",
        "precheck_l2_cap",
        "build_success",
        "eligible",
        "selected",
        "hard_learnable_score",
        "pad_strength",
        "pool_cleanliness",
        "val_richness",
        "target_pool_hidden_anomaly_ratio",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "failure_reason",
    ]
    for row in build_rows:
        if row["pair_id"] in selected_pair_names and row["eligible"]:
            best_entry = pair_best.get(row["pair_id"])
            row["selected"] = bool(best_entry and best_entry["row"]["shift_level"] == row["shift_level"])

    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_HARD_LEARNABLE,
            "dataset": "hai",
            "ranking_source": str(rankings_path),
            "ranking_notes": ranking_payload.get("notes", {}),
            "requested_topk": int(args.global_topk),
            "selected_count": int(len(manifest)),
            "methods_text": methods_text,
            "config": {
                "rule_shift_levels": shift_levels,
                "rule_target_pool_frac": args.hard_rule_target_pool_frac,
                "rule_val_frac": args.hard_rule_val_frac,
                "rule_guard": args.hard_rule_guard,
                "rule_search_step": args.hard_rule_search_step,
                "rule_max_pool_anom_ratio": args.hard_rule_max_pool_anom_ratio,
                "rule_min_target_pool": args.hard_rule_min_target_pool,
                "rule_min_val": args.hard_rule_min_val,
                "rule_min_test": args.hard_rule_min_test,
                "rule_min_anom_val": args.hard_rule_min_anom_val,
                "rule_min_anom_test": args.hard_rule_min_anom_test,
                "rule_min_pad_value": args.hard_rule_min_pad_value,
                "rule_max_pad_value": args.hard_rule_max_pad_value,
                "rule_val_count_ref": args.hard_rule_val_count_ref,
                "rule_val_anom_ref": args.hard_rule_val_anom_ref,
                "rule_max_precheck_l2_quantile": args.hard_rule_max_precheck_l2_quantile,
                "precheck_l2_cap": precheck_l2_cap,
                "allow_cross_version": bool(args.allow_cross_version),
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="HAI Cross-Entity Hard Learnable Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected hard-learnable pairs: {len(manifest)} / {args.global_topk}")
    if len(manifest) < args.global_topk:
        print("[WARN] Fewer eligible hard-learnable pairs than requested top-k.")


def build_hard_learnable_qband_pairs(args, entity_dirs, out_root: Path):
    if not args.rankings_json:
        raise ValueError("--rankings_json is required when --pair_rule is cross_entity_hard_learnable_qband")

    rankings_path = Path(args.rankings_json)
    if not rankings_path.exists():
        raise FileNotFoundError(rankings_path)

    entity_map = {entity_dir.name: entity_dir for entity_dir in entity_dirs}
    entity_meta = {entity_dir.name: read_entity_meta(entity_dir) for entity_dir in entity_dirs}
    protocol_dir = out_root / RULE_CROSS_ENTITY_HARD_LEARNABLE_QBAND
    protocol_dir.mkdir(parents=True, exist_ok=True)

    ranking_payload, ranked_rows = load_pad_pair_rankings(rankings_path)
    shift_levels = [s.strip() for s in args.hard_qband_shift_levels.split(",") if s.strip()]
    if not shift_levels:
        raise ValueError("--hard_qband_shift_levels must contain at least one shift level")
    if any(level != "hard" for level in shift_levels):
        raise ValueError("--hard_qband_shift_levels must be 'hard' for cross_entity_hard_learnable_qband")

    build_rows = []
    successful_rows = []
    pair_best = {}

    for ranking_row in ranked_rows:
        src_name = ranking_row["source_entity"]
        tgt_name = ranking_row["target_entity"]
        src_meta = entity_meta.get(src_name, {})
        tgt_meta = entity_meta.get(tgt_name, {})
        src_role = str(src_meta.get("split_role", "")).lower()
        tgt_role = str(tgt_meta.get("split_role", "")).lower()
        src_version = src_meta.get("version_tag") or src_name.split("-")[0]
        tgt_version = tgt_meta.get("version_tag") or tgt_name.split("-")[0]
        pair_name = ranking_row["pair_id"]
        precheck_l2 = safe_float(ranking_row.get("pad_feature_mean_l2"), default=float("nan"))
        base_row = {
            "global_pad_rank": ranking_row["global_pad_rank"],
            "source_entity": src_name,
            "target_entity": tgt_name,
            "pair_id": pair_name,
            "pad_value": ranking_row["pad_value"],
            "pad_domain_acc": ranking_row["pad_domain_acc"],
            "pad_domain_auc": ranking_row["pad_domain_auc"],
            "pad_feature_mean_l2": precheck_l2,
            "source_role": src_role,
            "target_role": tgt_role,
            "source_version": src_version,
            "target_version": tgt_version,
        }

        failure_reason = ""
        if src_name == tgt_name:
            failure_reason = "same_entity"
        elif src_name not in entity_map or tgt_name not in entity_map:
            failure_reason = "filtered_out_by_entity_subset"
        elif src_role != "train":
            failure_reason = "source_must_be_train"
        elif tgt_role != "test":
            failure_reason = "target_must_be_test"
        elif (not args.allow_cross_version) and tgt_version != src_version:
            failure_reason = "cross_version_blocked"

        if failure_reason:
            build_rows.append(
                {
                    **base_row,
                    "shift_level": "",
                    "build_success": False,
                    "eligible": False,
                    "selected": False,
                    "hard_learnable_qband_score": "",
                    "pad_strength": "",
                    "pool_cleanliness": "",
                    "val_richness": "",
                    "val_anomaly_ratio": "",
                    "target_pool_hidden_anomaly_ratio": "",
                    "train_normal_count": "",
                    "target_pool_count": "",
                    "val_count": "",
                    "test_count": "",
                    "val_anomaly_count": "",
                    "test_anomaly_count": "",
                    "failure_reason": failure_reason,
                }
            )
            continue

        for shift_level in shift_levels:
            source_dir = entity_map[src_name]
            target_dir = entity_map[tgt_name]
            out_dir = protocol_dir / f"{pair_name}__{shift_level}"
            ds_args = build_args(
                source_dir=source_dir,
                target_dir=target_dir,
                out_dir=out_dir,
                shift_level=shift_level,
                target_pool_frac=args.hard_qband_target_pool_frac,
                val_frac=args.hard_qband_val_frac,
                guard=args.hard_qband_guard,
                search_step=args.hard_qband_search_step,
                max_pool_anom_ratio=args.hard_qband_max_pool_anom_ratio_cap,
                min_target_pool=args.hard_qband_min_target_pool,
                min_val=args.hard_qband_min_val_floor,
                min_test=args.hard_qband_min_test,
                min_anom_val=args.hard_qband_min_val_anom_floor,
                min_anom_test=args.hard_qband_min_anom_test,
                seed=args.seed,
            )
            try:
                meta = create_dataset(ds_args)
                score_info = compute_hard_learnable_score(
                    pad_value=ranking_row["pad_value"],
                    target_pool_hidden_anomaly_ratio=meta.get("target_pool_hidden_anomaly_ratio"),
                    val_count=meta.get("val_count"),
                    val_anomaly_count=meta.get("val_anomaly_count"),
                    max_pool_anom_ratio=args.hard_qband_max_pool_anom_ratio_cap,
                    val_count_ref=max(args.hard_qband_min_val_floor, 1),
                    val_anom_ref=max(args.hard_qband_min_val_anom_floor, 1),
                    min_pad_value=args.hard_qband_min_pad_floor,
                    max_pad_value=2.0,
                )
                row = {
                    **base_row,
                    "shift_level": shift_level,
                    "build_success": True,
                    "eligible": False,
                    "selected": False,
                    "hard_learnable_qband_score": score_info["score"],
                    "pad_strength": score_info["components"]["pad_strength"],
                    "pool_cleanliness": score_info["components"]["pool_cleanliness"],
                    "val_richness": score_info["components"]["val_richness"],
                    "val_anomaly_ratio": safe_ratio(meta.get("val_anomaly_count"), meta.get("val_count")),
                    "target_pool_hidden_anomaly_ratio": meta["target_pool_hidden_anomaly_ratio"],
                    "train_normal_count": meta["train_normal_count"],
                    "target_pool_count": meta["target_pool_count"],
                    "val_count": meta["val_count"],
                    "test_count": meta["test_count"],
                    "val_anomaly_count": meta["val_anomaly_count"],
                    "test_anomaly_count": meta["test_anomaly_count"],
                    "failure_reason": "",
                }
                build_rows.append(row)
                successful_rows.append(row)
                pair_best.setdefault(pair_name, [])
                pair_best[pair_name].append({"row": row, "meta": meta, "score_info": score_info, "ranking_row": ranking_row})
            except Exception as exc:
                build_rows.append(
                    {
                        **base_row,
                        "shift_level": shift_level,
                        "build_success": False,
                        "eligible": False,
                        "selected": False,
                        "hard_learnable_qband_score": "",
                        "pad_strength": "",
                        "pool_cleanliness": "",
                        "val_richness": "",
                        "val_anomaly_ratio": "",
                        "target_pool_hidden_anomaly_ratio": "",
                        "train_normal_count": "",
                        "target_pool_count": "",
                        "val_count": "",
                        "test_count": "",
                        "val_anomaly_count": "",
                        "test_anomaly_count": "",
                        "failure_reason": str(exc),
                    }
                )

    thresholds = compute_hard_learnable_qband_thresholds(
        rows=successful_rows,
        min_pad_quantile=args.hard_qband_min_pad_quantile,
        max_precheck_l2_quantile=args.hard_qband_max_precheck_l2_quantile,
        max_pool_anom_ratio_quantile=args.hard_qband_max_pool_anom_ratio_quantile,
        min_val_count_quantile=args.hard_qband_min_val_count_quantile,
        min_val_anom_quantile=args.hard_qband_min_val_anom_quantile,
        max_val_anom_ratio_quantile=args.hard_qband_max_val_anom_ratio_quantile,
        min_pad_floor=args.hard_qband_min_pad_floor,
        max_pool_anom_ratio_cap=args.hard_qband_max_pool_anom_ratio_cap,
        min_val_floor=args.hard_qband_min_val_floor,
        min_val_anom_floor=args.hard_qband_min_val_anom_floor,
        max_val_anom_ratio_cap=args.hard_qband_max_val_anom_ratio_cap,
    )
    methods_text = build_hard_learnable_qband_methods_text(
        min_pad_quantile=args.hard_qband_min_pad_quantile,
        max_precheck_l2_quantile=args.hard_qband_max_precheck_l2_quantile,
        max_pool_anom_ratio_quantile=args.hard_qband_max_pool_anom_ratio_quantile,
        min_val_count_quantile=args.hard_qband_min_val_count_quantile,
        min_val_anom_quantile=args.hard_qband_min_val_anom_quantile,
        max_val_anom_ratio_quantile=args.hard_qband_max_val_anom_ratio_quantile,
        thresholds=thresholds,
    )

    selected_candidates = {}
    for pair_name, items in pair_best.items():
        best_item = None
        best_sort_key = None
        for item in items:
            row = item["row"]
            precheck_l2 = safe_float(row.get("pad_feature_mean_l2"), default=float("nan"))
            precheck_cap = safe_float(thresholds.get("max_precheck_l2"), default=float("nan"))
            precheck_ok = True
            if np.isfinite(precheck_cap):
                precheck_ok = np.isfinite(precheck_l2) and precheck_l2 <= precheck_cap
            pad_value = safe_float(row.get("pad_value"), default=0.0)
            pool_anom_ratio = safe_float(row.get("target_pool_hidden_anomaly_ratio"), default=1.0)
            val_count = int(row.get("val_count", 0))
            val_anom_count = int(row.get("val_anomaly_count", 0))
            val_anom_ratio = safe_float(row.get("val_anomaly_ratio"), default=float("inf"))
            eligible = (
                pad_value >= safe_float(thresholds.get("min_pad_value"), default=1.0)
                and pool_anom_ratio <= safe_float(thresholds.get("max_pool_anom_ratio"), default=1.0)
                and val_count >= safe_int(thresholds.get("min_val_count"), default=0)
                and val_anom_count >= safe_int(thresholds.get("min_val_anomaly_count"), default=0)
                and val_anom_ratio <= safe_float(thresholds.get("max_val_anomaly_ratio"), default=1.0)
                and precheck_ok
            )
            row["eligible"] = bool(eligible)
            row["precheck_l2_cap"] = thresholds.get("max_precheck_l2")
            if not eligible:
                continue
            sort_key = (
                -int(row["global_pad_rank"]),
                val_anom_count,
                val_count,
                float(row["pool_cleanliness"]),
            )
            if best_sort_key is None or sort_key > best_sort_key:
                best_sort_key = sort_key
                best_item = item
        if best_item is not None:
            selected_candidates[pair_name] = {"sort_key": best_sort_key, **best_item}

    selected_entries = sorted(
        selected_candidates.values(),
        key=lambda item: item["sort_key"],
        reverse=True,
    )[: args.global_topk]

    manifest = []
    selected_pair_names = {entry["row"]["pair_id"] for entry in selected_entries}
    for selection_order, entry in enumerate(selected_entries, start=1):
        row = entry["row"]
        row["selected"] = True
        meta = entry["meta"]
        ranking_row = entry["ranking_row"]
        meta["candidate_pair_shift_precheck"] = ranking_row["raw_row"].get("pad_latent", {})
        meta["ranking_rule"] = RULE_CROSS_ENTITY_HARD_LEARNABLE_QBAND
        meta["ranking_source"] = str(rankings_path)
        meta["global_pad_rank"] = int(ranking_row["global_pad_rank"])
        meta["selection_order"] = selection_order
        meta["selected_shift_level"] = row["shift_level"]
        meta["hard_learnable_qband"] = {
            **entry["score_info"],
            "precheck_feature_mean_l2": row["pad_feature_mean_l2"],
            "precheck_feature_mean_l2_cap": thresholds.get("max_precheck_l2"),
            "val_anomaly_ratio": row["val_anomaly_ratio"],
            "thresholds": thresholds,
            "methods_text": methods_text,
        }
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "shift_level",
        "source_role",
        "target_role",
        "source_version",
        "target_version",
        "pad_value",
        "pad_feature_mean_l2",
        "precheck_l2_cap",
        "build_success",
        "eligible",
        "selected",
        "hard_learnable_qband_score",
        "pad_strength",
        "pool_cleanliness",
        "val_richness",
        "val_anomaly_ratio",
        "target_pool_hidden_anomaly_ratio",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "failure_reason",
    ]
    for row in build_rows:
        if row["pair_id"] in selected_pair_names and row["eligible"]:
            chosen = selected_candidates.get(row["pair_id"])
            row["selected"] = bool(chosen and chosen["row"]["shift_level"] == row["shift_level"])

    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_HARD_LEARNABLE_QBAND,
            "dataset": "hai",
            "ranking_source": str(rankings_path),
            "ranking_notes": ranking_payload.get("notes", {}),
            "requested_topk": int(args.global_topk),
            "selected_count": int(len(manifest)),
            "methods_text": methods_text,
            "thresholds": thresholds,
            "config": {
                "rule_shift_levels": shift_levels,
                "rule_target_pool_frac": args.hard_qband_target_pool_frac,
                "rule_val_frac": args.hard_qband_val_frac,
                "rule_guard": args.hard_qband_guard,
                "rule_search_step": args.hard_qband_search_step,
                "rule_min_target_pool": args.hard_qband_min_target_pool,
                "rule_min_test": args.hard_qband_min_test,
                "rule_min_anom_test": args.hard_qband_min_anom_test,
                "rule_min_pad_quantile": args.hard_qband_min_pad_quantile,
                "rule_max_precheck_l2_quantile": args.hard_qband_max_precheck_l2_quantile,
                "rule_max_pool_anom_ratio_quantile": args.hard_qband_max_pool_anom_ratio_quantile,
                "rule_min_val_count_quantile": args.hard_qband_min_val_count_quantile,
                "rule_min_val_anom_quantile": args.hard_qband_min_val_anom_quantile,
                "rule_max_val_anom_ratio_quantile": args.hard_qband_max_val_anom_ratio_quantile,
                "rule_min_pad_floor": args.hard_qband_min_pad_floor,
                "rule_max_pool_anom_ratio_cap": args.hard_qband_max_pool_anom_ratio_cap,
                "rule_min_val_floor": args.hard_qband_min_val_floor,
                "rule_min_val_anom_floor": args.hard_qband_min_val_anom_floor,
                "rule_max_val_anom_ratio_cap": args.hard_qband_max_val_anom_ratio_cap,
                "allow_cross_version": bool(args.allow_cross_version),
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="HAI Cross-Entity Hard Learnable QBand Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected hard-learnable-qband pairs: {len(manifest)} / {args.global_topk}")
    if len(manifest) < args.global_topk:
        print("[WARN] Fewer eligible hard-learnable-qband pairs than requested top-k.")


def build_paper_safe_pairs(args, entity_dirs, out_root: Path):
    if not args.rankings_json:
        raise ValueError("--rankings_json is required when --pair_rule is cross_entity_paper_safe")

    rankings_path = Path(args.rankings_json)
    if not rankings_path.exists():
        raise FileNotFoundError(rankings_path)

    entity_map = {entity_dir.name: entity_dir for entity_dir in entity_dirs}
    entity_meta = {entity_dir.name: read_entity_meta(entity_dir) for entity_dir in entity_dirs}
    protocol_dir = out_root / RULE_CROSS_ENTITY_PAPER_SAFE
    protocol_dir.mkdir(parents=True, exist_ok=True)

    ranking_payload, ranked_rows = load_pad_pair_rankings(rankings_path)
    shift_levels = [s.strip() for s in args.paper_safe_shift_levels.split(",") if s.strip()]
    if not shift_levels:
        raise ValueError("--paper_safe_shift_levels must contain at least one shift level")

    build_rows = []
    successful_rows = []
    pair_best = {}

    for ranking_row in ranked_rows:
        src_name = ranking_row["source_entity"]
        tgt_name = ranking_row["target_entity"]
        src_meta = entity_meta.get(src_name, {})
        tgt_meta = entity_meta.get(tgt_name, {})
        src_role = str(src_meta.get("split_role", "")).lower()
        tgt_role = str(tgt_meta.get("split_role", "")).lower()
        src_version = src_meta.get("version_tag") or src_name.split("-")[0]
        tgt_version = tgt_meta.get("version_tag") or tgt_name.split("-")[0]
        pair_name = ranking_row["pair_id"]
        base_row = {
            "global_pad_rank": ranking_row["global_pad_rank"],
            "source_entity": src_name,
            "target_entity": tgt_name,
            "pair_id": pair_name,
            "pad_value": ranking_row["pad_value"],
            "pad_domain_acc": ranking_row["pad_domain_acc"],
            "pad_domain_auc": ranking_row["pad_domain_auc"],
            "pad_feature_mean_l2": ranking_row["pad_feature_mean_l2"],
            "source_role": src_role,
            "target_role": tgt_role,
            "source_version": src_version,
            "target_version": tgt_version,
        }

        failure_reason = ""
        if src_name == tgt_name:
            failure_reason = "same_entity"
        elif src_name not in entity_map or tgt_name not in entity_map:
            failure_reason = "filtered_out_by_entity_subset"
        elif src_role != "train":
            failure_reason = "source_must_be_train"
        elif tgt_role != "test":
            failure_reason = "target_must_be_test"
        elif (not args.allow_cross_version) and tgt_version != src_version:
            failure_reason = "cross_version_blocked"

        if failure_reason:
            build_rows.append(
                {
                    **base_row,
                    "shift_level": "",
                    "build_success": False,
                    "eligible": False,
                    "selected": False,
                    "val_anomaly_ratio": "",
                    "target_pool_hidden_anomaly_ratio": "",
                    "train_normal_count": "",
                    "target_pool_count": "",
                    "val_count": "",
                    "test_count": "",
                    "val_anomaly_count": "",
                    "test_anomaly_count": "",
                    "failure_reason": failure_reason,
                }
            )
            continue

        source_dir = entity_map[src_name]
        target_dir = entity_map[tgt_name]
        built = False
        for shift_index, shift_level in enumerate(shift_levels):
            out_dir = protocol_dir / f"{pair_name}__{shift_level}"
            ds_args = build_args(
                source_dir=source_dir,
                target_dir=target_dir,
                out_dir=out_dir,
                shift_level=shift_level,
                target_pool_frac=args.paper_safe_target_pool_frac,
                val_frac=args.paper_safe_val_frac,
                guard=args.paper_safe_guard,
                search_step=args.paper_safe_search_step,
                max_pool_anom_ratio=args.paper_safe_max_pool_anom_ratio_cap,
                min_target_pool=args.paper_safe_min_target_pool,
                min_val=args.paper_safe_min_val_floor,
                min_test=args.paper_safe_min_test,
                min_anom_val=args.paper_safe_min_val_anom_floor,
                min_anom_test=args.paper_safe_min_anom_test,
                seed=args.seed,
            )
            try:
                meta = create_dataset(ds_args)
                row = {
                    **base_row,
                    "shift_level": shift_level,
                    "shift_preference_rank": shift_index,
                    "build_success": True,
                    "eligible": False,
                    "selected": False,
                    "val_anomaly_ratio": safe_ratio(meta.get("val_anomaly_count"), meta.get("val_count")),
                    "target_pool_hidden_anomaly_ratio": meta["target_pool_hidden_anomaly_ratio"],
                    "train_normal_count": meta["train_normal_count"],
                    "target_pool_count": meta["target_pool_count"],
                    "val_count": meta["val_count"],
                    "test_count": meta["test_count"],
                    "val_anomaly_count": meta["val_anomaly_count"],
                    "test_anomaly_count": meta["test_anomaly_count"],
                    "failure_reason": "",
                }
                build_rows.append(row)
                successful_rows.append(row)
                pair_best[pair_name] = {"row": row, "meta": meta, "ranking_row": ranking_row}
                built = True
                break
            except Exception as exc:
                build_rows.append(
                    {
                        **base_row,
                        "shift_level": shift_level,
                        "shift_preference_rank": shift_index,
                        "build_success": False,
                        "eligible": False,
                        "selected": False,
                        "val_anomaly_ratio": "",
                        "target_pool_hidden_anomaly_ratio": "",
                        "train_normal_count": "",
                        "target_pool_count": "",
                        "val_count": "",
                        "test_count": "",
                        "val_anomaly_count": "",
                        "test_anomaly_count": "",
                        "failure_reason": str(exc),
                    }
                )
        if not built and pair_name not in pair_best:
            continue

    thresholds = compute_paper_safe_thresholds(
        rows=successful_rows,
        min_pad_quantile=args.paper_safe_min_pad_quantile,
        max_pool_anom_ratio_quantile=args.paper_safe_max_pool_anom_ratio_quantile,
        min_val_count_quantile=args.paper_safe_min_val_count_quantile,
        min_val_anom_quantile=args.paper_safe_min_val_anom_quantile,
        max_val_anom_ratio_quantile=args.paper_safe_max_val_anom_ratio_quantile,
        min_pad_floor=args.paper_safe_min_pad_floor,
        max_pool_anom_ratio_cap=args.paper_safe_max_pool_anom_ratio_cap,
        min_val_floor=args.paper_safe_min_val_floor,
        min_val_anom_floor=args.paper_safe_min_val_anom_floor,
        max_val_anom_ratio_cap=args.paper_safe_max_val_anom_ratio_cap,
    )
    methods_text = build_paper_safe_methods_text(
        preferred_shift_levels=shift_levels,
        min_pad_quantile=args.paper_safe_min_pad_quantile,
        max_pool_anom_ratio_quantile=args.paper_safe_max_pool_anom_ratio_quantile,
        min_val_count_quantile=args.paper_safe_min_val_count_quantile,
        min_val_anom_quantile=args.paper_safe_min_val_anom_quantile,
        max_val_anom_ratio_quantile=args.paper_safe_max_val_anom_ratio_quantile,
        thresholds=thresholds,
    )

    selected_candidates = {}
    for pair_name, item in pair_best.items():
        row = item["row"]
        pad_value = safe_float(row.get("pad_value"), default=0.0)
        pool_anom_ratio = safe_float(row.get("target_pool_hidden_anomaly_ratio"), default=1.0)
        val_count = int(row.get("val_count", 0))
        val_anom_count = int(row.get("val_anomaly_count", 0))
        val_anom_ratio = safe_float(row.get("val_anomaly_ratio"), default=float("inf"))
        eligible = (
            pad_value >= safe_float(thresholds.get("min_pad_value"), default=1.0)
            and pool_anom_ratio <= safe_float(thresholds.get("max_pool_anom_ratio"), default=1.0)
            and val_count >= safe_int(thresholds.get("min_val_count"), default=0)
            and val_anom_count >= safe_int(thresholds.get("min_val_anomaly_count"), default=0)
            and val_anom_ratio <= safe_float(thresholds.get("max_val_anomaly_ratio"), default=1.0)
        )
        row["eligible"] = bool(eligible)
        if not eligible:
            continue
        selected_candidates[pair_name] = {
            "sort_key": (
                -int(row["global_pad_rank"]),
                val_anom_count,
                val_count,
                -pool_anom_ratio,
            ),
            **item,
        }

    selected_entries = sorted(
        selected_candidates.values(),
        key=lambda item: item["sort_key"],
        reverse=True,
    )
    if args.global_topk > 0:
        selected_entries = selected_entries[: args.global_topk]

    manifest = []
    selected_pair_names = {entry["row"]["pair_id"] for entry in selected_entries}
    for selection_order, entry in enumerate(selected_entries, start=1):
        row = entry["row"]
        row["selected"] = True
        meta = entry["meta"]
        ranking_row = entry["ranking_row"]
        meta["candidate_pair_shift_precheck"] = ranking_row["raw_row"].get("pad_latent", {})
        meta["ranking_rule"] = RULE_CROSS_ENTITY_PAPER_SAFE
        meta["ranking_source"] = str(rankings_path)
        meta["global_pad_rank"] = int(ranking_row["global_pad_rank"])
        meta["selection_order"] = selection_order
        meta["selected_shift_level"] = row["shift_level"]
        meta["paper_safe"] = {
            "thresholds": thresholds,
            "methods_text": methods_text,
            "preferred_shift_levels": shift_levels,
            "val_anomaly_ratio": row["val_anomaly_ratio"],
            "target_pool_hidden_anomaly_ratio": row["target_pool_hidden_anomaly_ratio"],
        }
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "shift_level",
        "shift_preference_rank",
        "source_role",
        "target_role",
        "source_version",
        "target_version",
        "pad_value",
        "pad_domain_acc",
        "pad_domain_auc",
        "build_success",
        "eligible",
        "selected",
        "val_anomaly_ratio",
        "target_pool_hidden_anomaly_ratio",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "failure_reason",
    ]
    for row in build_rows:
        if row["pair_id"] in selected_pair_names and row.get("eligible"):
            chosen = selected_candidates.get(row["pair_id"])
            row["selected"] = bool(chosen and chosen["row"]["shift_level"] == row["shift_level"])

    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_PAPER_SAFE,
            "dataset": "hai",
            "ranking_source": str(rankings_path),
            "ranking_notes": ranking_payload.get("notes", {}),
            "requested_topk": int(args.global_topk),
            "selected_count": int(len(manifest)),
            "methods_text": methods_text,
            "thresholds": thresholds,
            "config": {
                "preferred_shift_levels": shift_levels,
                "rule_target_pool_frac": args.paper_safe_target_pool_frac,
                "rule_val_frac": args.paper_safe_val_frac,
                "rule_guard": args.paper_safe_guard,
                "rule_search_step": args.paper_safe_search_step,
                "rule_min_target_pool": args.paper_safe_min_target_pool,
                "rule_min_test": args.paper_safe_min_test,
                "rule_min_anom_test": args.paper_safe_min_anom_test,
                "rule_min_pad_quantile": args.paper_safe_min_pad_quantile,
                "rule_max_pool_anom_ratio_quantile": args.paper_safe_max_pool_anom_ratio_quantile,
                "rule_min_val_count_quantile": args.paper_safe_min_val_count_quantile,
                "rule_min_val_anom_quantile": args.paper_safe_min_val_anom_quantile,
                "rule_max_val_anom_ratio_quantile": args.paper_safe_max_val_anom_ratio_quantile,
                "rule_min_pad_floor": args.paper_safe_min_pad_floor,
                "rule_max_pool_anom_ratio_cap": args.paper_safe_max_pool_anom_ratio_cap,
                "rule_min_val_floor": args.paper_safe_min_val_floor,
                "rule_min_val_anom_floor": args.paper_safe_min_val_anom_floor,
                "rule_max_val_anom_ratio_cap": args.paper_safe_max_val_anom_ratio_cap,
                "allow_cross_version": bool(args.allow_cross_version),
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="HAI Paper-Safe Cross-Entity Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected paper-safe pairs: {len(manifest)}")
    if args.global_topk > 0 and len(manifest) < args.global_topk:
        print("[WARN] Fewer eligible paper-safe pairs than requested top-k.")


def build_learnable_shift_val_rich_pairs(args, entity_dirs, out_root: Path):
    if not args.rankings_json:
        raise ValueError("--rankings_json is required when --pair_rule is cross_entity_learnable_shift_val_rich")

    rankings_path = Path(args.rankings_json)
    if not rankings_path.exists():
        raise FileNotFoundError(rankings_path)

    entity_map = {entity_dir.name: entity_dir for entity_dir in entity_dirs}
    entity_meta = {entity_dir.name: read_entity_meta(entity_dir) for entity_dir in entity_dirs}
    protocol_dir = out_root / RULE_CROSS_ENTITY_LEARNABLE_SHIFT_VAL_RICH
    protocol_dir.mkdir(parents=True, exist_ok=True)

    ranking_payload, ranked_rows = load_pad_pair_rankings(rankings_path)
    shift_levels = [s.strip() for s in args.rule_shift_levels.split(",") if s.strip()]
    pair_best = {}
    build_rows = []

    for ranking_row in ranked_rows:
        src_name = ranking_row["source_entity"]
        tgt_name = ranking_row["target_entity"]
        src_meta = entity_meta.get(src_name, {})
        tgt_meta = entity_meta.get(tgt_name, {})
        src_role = str(src_meta.get("split_role", "")).lower()
        tgt_role = str(tgt_meta.get("split_role", "")).lower()
        src_version = src_meta.get("version_tag") or src_name.split("-")[0]
        tgt_version = tgt_meta.get("version_tag") or tgt_name.split("-")[0]
        pair_name = ranking_row["pair_id"]
        base_row = {
            "global_pad_rank": ranking_row["global_pad_rank"],
            "source_entity": src_name,
            "target_entity": tgt_name,
            "pair_id": pair_name,
            "pad_value": ranking_row["pad_value"],
            "pad_domain_acc": ranking_row["pad_domain_acc"],
            "pad_domain_auc": ranking_row["pad_domain_auc"],
            "pad_feature_mean_l2": ranking_row["pad_feature_mean_l2"],
            "source_role": src_role,
            "target_role": tgt_role,
            "source_version": src_version,
            "target_version": tgt_version,
        }

        failure_reason = ""
        if src_name == tgt_name:
            failure_reason = "same_entity"
        elif src_name not in entity_map or tgt_name not in entity_map:
            failure_reason = "filtered_out_by_entity_subset"
        elif src_role != "train":
            failure_reason = "source_must_be_train"
        elif tgt_role != "test":
            failure_reason = "target_must_be_test"
        elif (not args.allow_cross_version) and tgt_version != src_version:
            failure_reason = "cross_version_blocked"

        if failure_reason:
            build_rows.append(
                {
                    **base_row,
                    "shift_level": "",
                    "build_success": False,
                    "eligible": False,
                    "selected": False,
                    "learnable_score": "",
                    "shift_strength": "",
                    "pool_cleanliness": "",
                    "val_richness": "",
                    "source_vs_pool_domain_auc": "",
                    "target_pool_hidden_anomaly_ratio": "",
                    "train_normal_count": "",
                    "target_pool_count": "",
                    "val_count": "",
                    "test_count": "",
                    "val_anomaly_count": "",
                    "test_anomaly_count": "",
                    "failure_reason": failure_reason,
                }
            )
            continue

        for shift_level in shift_levels:
            source_dir = entity_map[src_name]
            target_dir = entity_map[tgt_name]
            out_dir = protocol_dir / f"{pair_name}__{shift_level}"
            ds_args = build_args(
                source_dir=source_dir,
                target_dir=target_dir,
                out_dir=out_dir,
                shift_level=shift_level,
                target_pool_frac=args.rule_target_pool_frac,
                val_frac=args.rule_val_frac,
                guard=args.rule_guard,
                search_step=args.rule_search_step,
                max_pool_anom_ratio=args.rule_max_pool_anom_ratio,
                min_target_pool=args.rule_min_target_pool,
                min_val=args.rule_min_val,
                min_test=args.rule_min_test,
                min_anom_val=args.rule_min_anom_val,
                min_anom_test=args.rule_min_anom_test,
                seed=args.seed,
            )
            try:
                meta = create_dataset(ds_args)
                score_info = compute_learnable_shift_val_rich_score(
                    pad_value=ranking_row["pad_value"],
                    source_vs_pool_domain_auc=(meta.get("domain_shift", {}).get("source_vs_target_pool", {}).get("domain_auc")),
                    target_pool_hidden_anomaly_ratio=meta.get("target_pool_hidden_anomaly_ratio"),
                    val_count=meta.get("val_count"),
                    val_anomaly_count=meta.get("val_anomaly_count"),
                    max_pool_anom_ratio=args.rule_max_pool_anom_ratio,
                    val_count_ref=args.rule_val_count_ref,
                    val_anom_ref=args.rule_val_anom_ref,
                    min_pad_value=args.rule_min_pad_value,
                    max_pad_value=args.rule_max_pad_value,
                )
                source_pool_auc = safe_float(meta.get("domain_shift", {}).get("source_vs_target_pool", {}).get("domain_auc"), default=0.0)
                pad_value = safe_float(ranking_row["pad_value"], default=0.0)
                eligible = (
                    pad_value >= args.rule_min_pad_value
                    and source_pool_auc >= args.rule_min_source_pool_auc
                    and float(meta.get("target_pool_hidden_anomaly_ratio", 1.0)) <= args.rule_max_pool_anom_ratio
                    and int(meta.get("val_count", 0)) >= args.rule_min_val
                    and int(meta.get("test_count", 0)) >= args.rule_min_test
                    and int(meta.get("val_anomaly_count", 0)) >= args.rule_min_anom_val
                    and int(meta.get("test_anomaly_count", 0)) >= args.rule_min_anom_test
                )
                meta["candidate_pair_shift_precheck"] = ranking_row["raw_row"].get("pad_latent", {})
                meta["ranking_rule"] = RULE_CROSS_ENTITY_LEARNABLE_SHIFT_VAL_RICH
                meta["ranking_source"] = str(rankings_path)
                meta["global_pad_rank"] = int(ranking_row["global_pad_rank"])
                meta["learnable_shift_val_rich"] = score_info
                row = {
                    **base_row,
                    "shift_level": shift_level,
                    "build_success": True,
                    "eligible": bool(eligible),
                    "selected": False,
                    "learnable_score": score_info["score"],
                    "shift_strength": score_info["components"]["shift_strength"],
                    "pool_cleanliness": score_info["components"]["pool_cleanliness"],
                    "val_richness": score_info["components"]["val_richness"],
                    "source_vs_pool_domain_auc": source_pool_auc,
                    "target_pool_hidden_anomaly_ratio": meta["target_pool_hidden_anomaly_ratio"],
                    "train_normal_count": meta["train_normal_count"],
                    "target_pool_count": meta["target_pool_count"],
                    "val_count": meta["val_count"],
                    "test_count": meta["test_count"],
                    "val_anomaly_count": meta["val_anomaly_count"],
                    "test_anomaly_count": meta["test_anomaly_count"],
                    "failure_reason": "",
                }
                build_rows.append(row)
                if eligible:
                    current = pair_best.get(pair_name)
                    sort_key = (
                        float(score_info["score"]),
                        int(meta["val_anomaly_count"]),
                        int(meta["val_count"]),
                        pad_value,
                    )
                    if current is None or sort_key > current["sort_key"]:
                        pair_best[pair_name] = {
                            "sort_key": sort_key,
                            "row": row,
                            "meta": meta,
                        }
            except Exception as exc:
                build_rows.append(
                    {
                        **base_row,
                        "shift_level": shift_level,
                        "build_success": False,
                        "eligible": False,
                        "selected": False,
                        "learnable_score": "",
                        "shift_strength": "",
                        "pool_cleanliness": "",
                        "val_richness": "",
                        "source_vs_pool_domain_auc": "",
                        "target_pool_hidden_anomaly_ratio": "",
                        "train_normal_count": "",
                        "target_pool_count": "",
                        "val_count": "",
                        "test_count": "",
                        "val_anomaly_count": "",
                        "test_anomaly_count": "",
                        "failure_reason": str(exc),
                    }
                )

    selected_entries = sorted(
        pair_best.values(),
        key=lambda item: item["sort_key"],
        reverse=True,
    )[: args.global_topk]

    manifest = []
    selected_pair_names = {entry["row"]["pair_id"] for entry in selected_entries}
    for selection_order, entry in enumerate(selected_entries, start=1):
        entry["row"]["selected"] = True
        meta = entry["meta"]
        meta["selection_order"] = selection_order
        meta["selected_shift_level"] = entry["row"]["shift_level"]
        manifest.append(meta)

    build_columns = [
        "global_pad_rank",
        "source_entity",
        "target_entity",
        "pair_id",
        "shift_level",
        "source_role",
        "target_role",
        "source_version",
        "target_version",
        "pad_value",
        "build_success",
        "eligible",
        "selected",
        "learnable_score",
        "shift_strength",
        "pool_cleanliness",
        "val_richness",
        "source_vs_pool_domain_auc",
        "target_pool_hidden_anomaly_ratio",
        "train_normal_count",
        "target_pool_count",
        "val_count",
        "test_count",
        "val_anomaly_count",
        "test_anomaly_count",
        "failure_reason",
    ]
    for row in build_rows:
        if row["pair_id"] in selected_pair_names and row["eligible"]:
            best_entry = pair_best.get(row["pair_id"])
            row["selected"] = bool(best_entry and best_entry["row"]["shift_level"] == row["shift_level"])

    save_json(
        protocol_dir / "selection_summary.json",
        {
            "rule_name": RULE_CROSS_ENTITY_LEARNABLE_SHIFT_VAL_RICH,
            "dataset": "hai",
            "ranking_source": str(rankings_path),
            "ranking_notes": ranking_payload.get("notes", {}),
            "requested_topk": int(args.global_topk),
            "selected_count": int(len(manifest)),
            "config": {
                "rule_shift_levels": shift_levels,
                "rule_target_pool_frac": args.rule_target_pool_frac,
                "rule_val_frac": args.rule_val_frac,
                "rule_guard": args.rule_guard,
                "rule_search_step": args.rule_search_step,
                "rule_max_pool_anom_ratio": args.rule_max_pool_anom_ratio,
                "rule_min_target_pool": args.rule_min_target_pool,
                "rule_min_val": args.rule_min_val,
                "rule_min_test": args.rule_min_test,
                "rule_min_anom_val": args.rule_min_anom_val,
                "rule_min_anom_test": args.rule_min_anom_test,
                "rule_min_pad_value": args.rule_min_pad_value,
                "rule_min_source_pool_auc": args.rule_min_source_pool_auc,
                "rule_val_count_ref": args.rule_val_count_ref,
                "rule_val_anom_ref": args.rule_val_anom_ref,
                "allow_cross_version": bool(args.allow_cross_version),
            },
            "rows": public_rows(build_rows),
        },
    )
    write_table_csv(protocol_dir / "selection_summary.csv", build_rows, build_columns)
    write_table_markdown(
        protocol_dir / "selection_summary.md",
        build_rows,
        build_columns,
        title="HAI Learnable Shift Val-Rich Selection",
    )

    manifest_path = protocol_dir / "manifest.json"
    save_json(manifest_path, manifest)
    print(f"[DONE] Saved manifest: {manifest_path}")
    print(f"[DONE] Selected learnable val-rich pairs: {len(manifest)} / {args.global_topk}")
    if len(manifest) < args.global_topk:
        print("[WARN] Fewer eligible learnable val-rich pairs than requested top-k.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/hai")
    ap.add_argument("--out_root", default="data/hai_experiments")
    ap.add_argument("--entities", default=None, help="Comma-separated cached HAI entities to include.")
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--shift_levels", default="auto,hard")
    ap.add_argument("--topk_cross", type=int, default=1)
    ap.add_argument("--target_pool_frac", type=float, default=DEFAULT_TARGET_POOL_FRAC)
    ap.add_argument("--val_frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--guard", type=int, default=DEFAULT_GUARD)
    ap.add_argument("--search_step", type=int, default=DEFAULT_SEARCH_STEP)
    ap.add_argument("--max_pool_anom_ratio", type=float, default=DEFAULT_MAX_POOL_ANOM_RATIO)
    ap.add_argument("--min_target_pool", type=int, default=DEFAULT_MIN_TARGET_POOL)
    ap.add_argument("--min_val", type=int, default=DEFAULT_MIN_VAL)
    ap.add_argument("--min_test", type=int, default=DEFAULT_MIN_TEST)
    ap.add_argument("--min_anom_val", type=int, default=DEFAULT_MIN_ANOM_VAL)
    ap.add_argument("--min_anom_test", type=int, default=DEFAULT_MIN_ANOM_TEST)
    ap.add_argument("--min_target_anom", type=int, default=3)
    ap.add_argument("--allow_cross_version", action="store_true")
    ap.add_argument(
        "--pair_rule",
        default="per_source_topk",
        choices=[
            "per_source_topk",
            RULE_CROSS_ENTITY_LEARNABLE_SHIFT_VAL_RICH,
            RULE_CROSS_ENTITY_HARD_LEARNABLE,
            RULE_CROSS_ENTITY_HARD_LEARNABLE_QBAND,
            RULE_CROSS_ENTITY_PAPER_SAFE,
        ],
    )
    ap.add_argument("--rankings_json", default=None)
    ap.add_argument("--global_topk", type=int, default=3)
    ap.add_argument("--rule_shift_levels", default="medium,hard")
    ap.add_argument("--rule_target_pool_frac", type=float, default=0.20)
    ap.add_argument("--rule_val_frac", type=float, default=0.45)
    ap.add_argument("--rule_guard", type=int, default=0)
    ap.add_argument("--rule_search_step", type=int, default=DEFAULT_SEARCH_STEP)
    ap.add_argument("--rule_max_pool_anom_ratio", type=float, default=0.05)
    ap.add_argument("--rule_min_target_pool", type=int, default=64)
    ap.add_argument("--rule_min_val", type=int, default=128)
    ap.add_argument("--rule_min_test", type=int, default=192)
    ap.add_argument("--rule_min_anom_val", type=int, default=8)
    ap.add_argument("--rule_min_anom_test", type=int, default=10)
    ap.add_argument("--rule_min_pad_value", type=float, default=1.0)
    ap.add_argument("--rule_max_pad_value", type=float, default=2.0)
    ap.add_argument("--rule_min_source_pool_auc", type=float, default=0.90)
    ap.add_argument("--rule_val_count_ref", type=int, default=160)
    ap.add_argument("--rule_val_anom_ref", type=int, default=16)
    ap.add_argument("--hard_rule_shift_levels", default="hard")
    ap.add_argument("--hard_rule_target_pool_frac", type=float, default=0.20)
    ap.add_argument("--hard_rule_val_frac", type=float, default=0.45)
    ap.add_argument("--hard_rule_guard", type=int, default=0)
    ap.add_argument("--hard_rule_search_step", type=int, default=DEFAULT_SEARCH_STEP)
    ap.add_argument("--hard_rule_max_pool_anom_ratio", type=float, default=0.10)
    ap.add_argument("--hard_rule_min_target_pool", type=int, default=DEFAULT_MIN_TARGET_POOL)
    ap.add_argument("--hard_rule_min_val", type=int, default=32)
    ap.add_argument("--hard_rule_min_test", type=int, default=DEFAULT_MIN_TEST)
    ap.add_argument("--hard_rule_min_anom_val", type=int, default=7)
    ap.add_argument("--hard_rule_min_anom_test", type=int, default=DEFAULT_MIN_ANOM_TEST)
    ap.add_argument("--hard_rule_min_pad_value", type=float, default=1.0)
    ap.add_argument("--hard_rule_max_pad_value", type=float, default=2.0)
    ap.add_argument("--hard_rule_val_count_ref", type=int, default=64)
    ap.add_argument("--hard_rule_val_anom_ref", type=int, default=12)
    ap.add_argument("--hard_rule_max_precheck_l2_quantile", type=float, default=0.80)
    ap.add_argument("--hard_qband_shift_levels", default="hard")
    ap.add_argument("--hard_qband_target_pool_frac", type=float, default=0.20)
    ap.add_argument("--hard_qband_val_frac", type=float, default=0.45)
    ap.add_argument("--hard_qband_guard", type=int, default=0)
    ap.add_argument("--hard_qband_search_step", type=int, default=DEFAULT_SEARCH_STEP)
    ap.add_argument("--hard_qband_min_target_pool", type=int, default=DEFAULT_MIN_TARGET_POOL)
    ap.add_argument("--hard_qband_min_test", type=int, default=DEFAULT_MIN_TEST)
    ap.add_argument("--hard_qband_min_anom_test", type=int, default=DEFAULT_MIN_ANOM_TEST)
    ap.add_argument("--hard_qband_min_pad_quantile", type=float, default=0.50)
    ap.add_argument("--hard_qband_max_precheck_l2_quantile", type=float, default=0.80)
    ap.add_argument("--hard_qband_max_pool_anom_ratio_quantile", type=float, default=0.80)
    ap.add_argument("--hard_qband_min_val_count_quantile", type=float, default=0.30)
    ap.add_argument("--hard_qband_min_val_anom_quantile", type=float, default=0.20)
    ap.add_argument("--hard_qband_max_val_anom_ratio_quantile", type=float, default=0.80)
    ap.add_argument("--hard_qband_min_pad_floor", type=float, default=1.0)
    ap.add_argument("--hard_qband_max_pool_anom_ratio_cap", type=float, default=0.10)
    ap.add_argument("--hard_qband_min_val_floor", type=int, default=32)
    ap.add_argument("--hard_qband_min_val_anom_floor", type=int, default=7)
    ap.add_argument("--hard_qband_max_val_anom_ratio_cap", type=float, default=0.40)
    ap.add_argument("--paper_safe_shift_levels", default="hard,medium")
    ap.add_argument("--paper_safe_target_pool_frac", type=float, default=0.20)
    ap.add_argument("--paper_safe_val_frac", type=float, default=0.45)
    ap.add_argument("--paper_safe_guard", type=int, default=0)
    ap.add_argument("--paper_safe_search_step", type=int, default=DEFAULT_SEARCH_STEP)
    ap.add_argument("--paper_safe_min_target_pool", type=int, default=DEFAULT_MIN_TARGET_POOL)
    ap.add_argument("--paper_safe_min_test", type=int, default=DEFAULT_MIN_TEST)
    ap.add_argument("--paper_safe_min_anom_test", type=int, default=DEFAULT_MIN_ANOM_TEST)
    ap.add_argument("--paper_safe_min_pad_quantile", type=float, default=0.50)
    ap.add_argument("--paper_safe_max_pool_anom_ratio_quantile", type=float, default=0.80)
    ap.add_argument("--paper_safe_min_val_count_quantile", type=float, default=0.30)
    ap.add_argument("--paper_safe_min_val_anom_quantile", type=float, default=0.20)
    ap.add_argument("--paper_safe_max_val_anom_ratio_quantile", type=float, default=0.80)
    ap.add_argument("--paper_safe_min_pad_floor", type=float, default=1.0)
    ap.add_argument("--paper_safe_max_pool_anom_ratio_cap", type=float, default=0.10)
    ap.add_argument("--paper_safe_min_val_floor", type=int, default=32)
    ap.add_argument("--paper_safe_min_val_anom_floor", type=int, default=7)
    ap.add_argument("--paper_safe_max_val_anom_ratio_cap", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    shift_levels = [s.strip() for s in args.shift_levels.split(",") if s.strip()]
    entity_dirs = list_entity_dirs(data_root)
    if not entity_dirs:
        raise FileNotFoundError(f"No cached HAI entity folders with source.npz/target.npz under {data_root}")

    if args.entities:
        keep = {e.strip() for e in args.entities.split(",") if e.strip()}
        entity_dirs = [p for p in entity_dirs if p.name in keep]
    if args.max_entities > 0:
        entity_dirs = entity_dirs[: args.max_entities]
    if not entity_dirs:
        raise ValueError("No HAI entities left after filtering.")

    if args.pair_rule == RULE_CROSS_ENTITY_LEARNABLE_SHIFT_VAL_RICH:
        build_learnable_shift_val_rich_pairs(args, entity_dirs, out_root)
        return
    if args.pair_rule == RULE_CROSS_ENTITY_HARD_LEARNABLE:
        build_hard_learnable_pairs(args, entity_dirs, out_root)
        return
    if args.pair_rule == RULE_CROSS_ENTITY_HARD_LEARNABLE_QBAND:
        build_hard_learnable_qband_pairs(args, entity_dirs, out_root)
        return
    if args.pair_rule == RULE_CROSS_ENTITY_PAPER_SAFE:
        build_paper_safe_pairs(args, entity_dirs, out_root)
        return

    source_entities = [p for p in entity_dirs if str(read_entity_meta(p).get("split_role", "")).lower() == "train"]
    target_entities = [p for p in entity_dirs if str(read_entity_meta(p).get("split_role", "")).lower() == "test"]
    if not source_entities or not target_entities:
        raise ValueError("HAI builder expects at least one train entity and one test entity.")

    same_version_only = not args.allow_cross_version
    manifest = []
    for shift_level in shift_levels:
        for source_dir in source_entities:
            ranked = rank_cross_targets(
                source_dir,
                candidate_targets(source_dir, target_entities, same_version_only),
                min_target_anom=args.min_target_anom,
            )
            for _, _, target_dir, shift in ranked[: args.topk_cross]:
                out_dir = out_root / f"cross_entity_{shift_level}" / f"{source_dir.name}__to__{target_dir.name}"
                ds_args = build_args(
                    source_dir=source_dir,
                    target_dir=target_dir,
                    out_dir=out_dir,
                    shift_level=shift_level,
                    target_pool_frac=args.target_pool_frac,
                    val_frac=args.val_frac,
                    guard=args.guard,
                    search_step=args.search_step,
                    max_pool_anom_ratio=args.max_pool_anom_ratio,
                    min_target_pool=args.min_target_pool,
                    min_val=args.min_val,
                    min_test=args.min_test,
                    min_anom_val=args.min_anom_val,
                    min_anom_test=args.min_anom_test,
                    seed=args.seed,
                )
                try:
                    meta = create_dataset(ds_args)
                    meta["candidate_pair_shift_precheck"] = shift
                    meta["same_version_only"] = bool(same_version_only)
                    manifest.append(meta)
                except Exception as exc:
                    print(f"[WARN] cross {shift_level} {source_dir.name}->{target_dir.name}: {exc}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved manifest with {len(manifest)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
