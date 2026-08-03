import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adaptnas.search_space import ArchConfig
from src.pipeline import (
    TSTrainer,
    binarize_y,
    infer_cached_pretrain_root,
    load_all_cached_entities_for_pretrain,
    load_npz_if_exists,
    run_final_only_option2,
    set_global_seed,
    split_source_holdout_normal,
)
from src.ts_tcc.config_files.HAR_Configs import Config
from src.ts_tcc.dataloader.dataloader import Load_Dataset
from src.ts_tcc.models.TC import TC
from src.ts_tcc.models.model import base_Model


def parse_args():
    ap = argparse.ArgumentParser(
        description="Analyze how well combined-mode search upper_obj aligns with candidate val_mixed AUROC."
    )
    ap.add_argument("--benchmark_root", required=True, help="Benchmark folder containing summary.json and pair subdirs.")
    ap.add_argument(
        "--output_root",
        default=None,
        help="Output folder for candidate-level alignment reports. Defaults to <benchmark_root>/nas_search_alignment.",
    )
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument(
        "--topk",
        default="1,3,5",
        help="Comma-separated k values for top-k hit rate, based on search ranking by lower upper_obj.",
    )
    ap.add_argument(
        "--pair_ids",
        default="",
        help="Optional comma-separated pair ids to analyze. Empty means all pairs from summary.json.",
    )
    ap.add_argument("--force", action="store_true", help="Recompute cached candidate evaluations.")
    ap.add_argument(
        "--combined_final_candidate_warmup_steps",
        type=int,
        default=80,
        help="Final-only warmup steps used in candidate reruns. Keep aligned with pipeline defaults unless overridden.",
    )
    ap.add_argument(
        "--combined_final_steps",
        type=int,
        default=200,
        help="Final-only bilevel steps used in candidate reruns. Keep aligned with pipeline defaults unless overridden.",
    )
    ap.add_argument(
        "--combined_final_patience",
        type=int,
        default=10,
        help="Early-stop patience used in candidate reruns. Keep aligned with pipeline defaults unless overridden.",
    )
    return ap.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_arch(arch_text: str) -> ArchConfig:
    return eval(arch_text, {"__builtins__": {}}, {"ArchConfig": ArchConfig})


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def pair_id_from_manifest_row(row):
    return f"{row['source_entity']}__to__{row['target_entity']}"


def build_args_namespace(summary, result_json, cli_args):
    settings = summary.get("settings") or {}
    oneclass = result_json.get("oneclass") or {}
    search_cfg = dict(oneclass.get("search_config") or {})
    final_cfg = dict(oneclass.get("final_config") or {})

    def cfg(name, default):
        return search_cfg.get(name, default)

    args = SimpleNamespace()
    args.seed = int(summary.get("seed", 42))
    args.batch_size = int(settings.get("batch_size", 64))
    args.oneclass_method = str(
        settings.get(
            "oneclass_method",
            oneclass.get("search_method", oneclass.get("method", "deepsvdd")),
        )
    )
    args.weighting_oneclass_method = settings.get(
        "weighting_oneclass_method",
        oneclass.get("weighting_method"),
    )
    args.final_oneclass_method = settings.get(
        "final_oneclass_method",
        oneclass.get("final_method"),
    )
    args.oneclass_epochs = int(settings.get("oneclass_epochs", search_cfg.get("epochs", 10)))
    args.oneclass_final_epochs = int(settings.get("oneclass_final_epochs", final_cfg.get("epochs", 20)))
    args.oneclass_lr = float(search_cfg.get("lr", 1e-3))
    args.oneclass_batch_size = int(settings.get("oneclass_batch_size", search_cfg.get("batch_size", 1024)))
    args.oneclass_max_fit = int(settings.get("oneclass_max_fit", search_cfg.get("max_fit", 5000) or 0))
    args.knn_k = int(cfg("knn_k", 5))
    args.ocsvm_nu = float(cfg("ocsvm_nu", 0.05))
    args.ocsvm_kernel = str(cfg("ocsvm_kernel", "rbf"))
    args.ocsvm_gamma = str(cfg("ocsvm_gamma", "scale"))
    args.ocsvm_degree = int(cfg("ocsvm_degree", 3))
    args.ocsvm_coef0 = float(cfg("ocsvm_coef0", 0.0))
    args.svdd_hidden_dim = int(cfg("svdd_hidden_dim", 128))
    args.svdd_rep_dim = int(cfg("svdd_rep_dim", 64))
    args.svdd_nu = float(cfg("svdd_nu", 0.05))
    args.svdd_warmup_epochs = int(search_cfg.get("svdd_warmup_epochs", 2))
    args.svdd_final_warmup_epochs = int(final_cfg.get("svdd_warmup_epochs", 5))
    args.ae_hidden_dim = int(cfg("ae_hidden_dim", 128))
    args.ae_latent_dim = int(cfg("ae_latent_dim", 64))
    args.maha_hidden_dim = int(cfg("maha_hidden_dim", 128))
    args.maha_rep_dim = int(cfg("maha_rep_dim", 64))
    args.maha_shrinkage = float(cfg("maha_shrinkage", 1e-2))
    args.gmm_hidden_dim = int(cfg("gmm_hidden_dim", 128))
    args.gmm_rep_dim = int(cfg("gmm_rep_dim", 64))
    args.gmm_components = int(cfg("gmm_components", 3))
    args.gmm_covariance_type = str(cfg("gmm_covariance_type", "diag"))
    args.gmm_reg_covar = float(cfg("gmm_reg_covar", 1e-4))
    args.gmm_warmup_epochs = int(cfg("gmm_warmup_epochs", 2))
    args.proto_hidden_dim = int(cfg("proto_hidden_dim", 128))
    args.proto_rep_dim = int(cfg("proto_rep_dim", 64))
    args.proto_count = int(cfg("proto_count", 4))
    args.proto_separation_weight = float(cfg("proto_separation_weight", 0.1))
    args.proto_separation_margin = float(cfg("proto_separation_margin", 1.0))
    args.combined_upper_gap = float(settings.get("combined_upper_gap", 1.0))
    args.combined_final_candidate_warmup_steps = int(cli_args.combined_final_candidate_warmup_steps)
    args.combined_final_steps = int(cli_args.combined_final_steps)
    args.combined_final_patience = int(cli_args.combined_final_patience)
    return args


def prepare_tstcc_backbone(case_row, args, device):
    split_dir = REPO_ROOT / case_row["out_dir"]
    train_path = split_dir / "train_normal.npz"
    target_pool_path = split_dir / "target_pool_unlabeled.npz"

    X_train_norm, _ = load_npz_if_exists(train_path)
    X_target_pool, _ = load_npz_if_exists(target_pool_path)
    in_ch = int(X_train_norm.shape[-1])
    norm_path = str(train_path).replace("\\", "/")

    X_pretrain_multi = None
    pretrain_multi_name = None
    candidate_root = infer_cached_pretrain_root(norm_path)
    if candidate_root and Path(candidate_root).is_dir():
        try:
            cached_name = Path(candidate_root).name.lower()
            pretrain_entity_prefix = None
            pretrain_entity_cap = 0
            if cached_name in {"exathlon", "hai"}:
                source_entity = case_row["source_entity"]
                pretrain_entity_prefix = source_entity.split("-")[0] if "-" in source_entity else source_entity
                pretrain_entity_cap = 16 if cached_name == "exathlon" else 256
            X_pretrain_multi = load_all_cached_entities_for_pretrain(
                candidate_root,
                window=128,
                in_channels=in_ch,
                entity_prefix=pretrain_entity_prefix,
                max_windows_per_entity=pretrain_entity_cap,
                seed=args.seed,
            )
            pretrain_multi_name = Path(candidate_root).name
        except RuntimeError:
            X_pretrain_multi = None
            pretrain_multi_name = None

    if X_pretrain_multi is not None:
        train_x = X_pretrain_multi
        label = pretrain_multi_name if pretrain_multi_name is not None else "cached entities"
        print(f"[ALIGN] TS-TCC pretraining on multi-entity {label}: {train_x.shape[0]} windows.")
    else:
        train_x = np.concatenate([X_train_norm, X_target_pool], axis=0)
        print(f"[ALIGN] TS-TCC pretraining on train_normal + target_pool_unlabeled: {train_x.shape[0]} windows.")

    if len(train_x) < 2:
        raise ValueError(f"TS-TCC pretraining requires at least 2 windows, got {len(train_x)}.")

    train_ss = {
        "samples": torch.tensor(train_x, dtype=torch.float32),
        "labels": torch.zeros(len(train_x)),
    }

    config = Config()
    config.input_channels = in_ch
    if hasattr(config, "input_length"):
        config.input_length = 128
    if hasattr(config, "num_classes"):
        config.num_classes = 2
    ssl_batch_size = max(2, min(int(args.batch_size), int(len(train_x))))
    config.batch_size = ssl_batch_size
    config.num_epoch = int(args.epochs_pretrain)

    model = base_Model(config).to(device)
    temporal_contr_model = TC(config, device).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=3e-4)
    temp_opt = torch.optim.Adam(temporal_contr_model.parameters(), lr=config.lr, weight_decay=3e-4)
    tstcc = TSTrainer(model, temporal_contr_model, model_opt, temp_opt, device, config)

    train_dataset = Load_Dataset(train_ss, config, training_mode="self_supervised")
    train_loader = DataLoader(
        train_dataset,
        batch_size=ssl_batch_size,
        shuffle=True,
        drop_last=True,
    )
    tstcc.train(train_dl=train_loader, training_mode="self_supervised")
    model.eval()
    return model


def load_case_arrays(case_row, seed):
    split_dir = REPO_ROOT / case_row["out_dir"]
    X_train_norm, _ = load_npz_if_exists(split_dir / "train_normal.npz")
    X_target_pool, _ = load_npz_if_exists(split_dir / "target_pool_unlabeled.npz")
    X_val, y_val = load_npz_if_exists(split_dir / "val_mixed.npz")
    y_val = binarize_y(y_val)
    Xs_train, Xs_holdout = split_source_holdout_normal(X_train_norm, holdout_ratio=0.2, seed=seed)
    Ys_source = np.zeros(len(Xs_train), dtype=int)
    return {
        "split_dir": split_dir,
        "X_train_norm": X_train_norm,
        "X_target_pool": X_target_pool,
        "X_val": X_val,
        "y_val": y_val,
        "Xs_train": Xs_train,
        "Xs_holdout": Xs_holdout,
        "Ys_source": Ys_source,
        "in_ch": int(X_train_norm.shape[-1]),
        "N_ITERS": 3,
    }


def rank_with_average_ties(values, reverse=False):
    pairs = sorted(enumerate(values), key=lambda x: (-x[1], x[0]) if reverse else (x[1], x[0]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(xs, ys):
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x_std = x.std()
    y_std = y.std()
    if x_std == 0 or y_std == 0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).mean() / (x_std * y_std))


def spearman_corr(xs, ys):
    if len(xs) < 2:
        return None
    xr = rank_with_average_ties(list(xs), reverse=False)
    yr = rank_with_average_ties(list(ys), reverse=False)
    return pearson_corr(xr, yr)


def metric_or_none(metrics, key):
    if not metrics:
        return None
    value = metrics.get(key)
    return None if value is None else float(value)


def unique_candidates_from_search(result_json):
    grouped = {}
    for entry in result_json.get("search_history") or []:
        arch = entry.get("arch")
        if not arch:
            continue
        rec = grouped.setdefault(
            arch,
            {
                "arch": arch,
                "upper_obj_min": None,
                "occurrences": 0,
                "iters": [],
                "families": set(),
            },
        )
        upper_obj = float(entry["upper_obj"])
        rec["upper_obj_min"] = upper_obj if rec["upper_obj_min"] is None else min(rec["upper_obj_min"], upper_obj)
        rec["occurrences"] += 1
        rec["iters"].append(int(entry.get("iter", 0)))
        family = entry.get("search_family")
        if family:
            rec["families"].add(family)
    rows = []
    for rec in grouped.values():
        rows.append(
            {
                "arch": rec["arch"],
                "upper_obj_min": float(rec["upper_obj_min"]),
                "occurrences": int(rec["occurrences"]),
                "iters": sorted(rec["iters"]),
                "families": ",".join(sorted(rec["families"])) if rec["families"] else "",
            }
        )
    rows.sort(key=lambda r: (r["upper_obj_min"], r["arch"]))
    return rows


def evaluate_candidate_arch(
    arch_text,
    arrays,
    args,
    device,
    tstcc_backbone,
    case_out_dir,
):
    arch_cfg = parse_arch(arch_text)
    out = run_final_only_option2(
        arch_name="candidate_eval",
        arch_cfg=arch_cfg,
        Xs=arrays["Xs_train"],
        Ys=arrays["Ys_source"],
        X_target_pool=arrays["X_target_pool"],
        X_source_holdout=arrays["Xs_holdout"],
        X_select=arrays["X_val"],
        Y_select=arrays["y_val"],
        X_report=arrays["X_val"],
        Y_report=arrays["y_val"],
        args=args,
        device=device,
        in_ch=arrays["in_ch"],
        N_ITERS=arrays["N_ITERS"],
        tstcc_backbone=tstcc_backbone,
        seed=args.seed,
        out_dir=str(case_out_dir),
        selection_name="val_mixed",
        report_name="val_mixed",
    )
    selection = out.get("selection_metrics_uad") or {}
    return {
        "arch": arch_text,
        "selection_metrics_uad": selection,
        "val_auroc": metric_or_none(selection, "auroc"),
        "val_auprc": metric_or_none(selection, "auprc"),
        "val_f1_best": metric_or_none(selection, "f1_best"),
    }


def evaluate_case(case_row, summary, benchmark_root, output_root, args, topk_values):
    pair_id = pair_id_from_manifest_row(case_row)
    result_path = benchmark_root / pair_id / "adaptnas_combined_results.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)

    result_json = load_json(result_path)
    candidate_rows = unique_candidates_from_search(result_json)
    if not candidate_rows:
        return None, []

    case_out_dir = output_root / pair_id
    cache_path = case_out_dir / "candidate_eval_cache.json"
    cache = {}
    if cache_path.exists() and not args.force:
        cache = load_json(cache_path)

    required_arches = [cand["arch"] for cand in candidate_rows]
    missing_arches = [arch for arch in required_arches if arch not in cache or args.force]

    arg_ns = build_args_namespace(summary, result_json, args)
    arg_ns.epochs_pretrain = int((summary.get("settings") or {}).get("epochs_pretrain", 10))

    device = resolve_device(args.device if args.device != "auto" else summary.get("device", "auto"))
    arrays = load_case_arrays(case_row, seed=arg_ns.seed)
    arrays["N_ITERS"] = max(int(e.get("iter", 0)) for e in (result_json.get("search_history") or []) if e.get("iter") is not None)
    tstcc_backbone = None
    if missing_arches:
        set_global_seed(arg_ns.seed)
        print(f"[ALIGN] {pair_id}: pretraining TS-TCC once, then evaluating {len(missing_arches)} / {len(candidate_rows)} uncached candidates.")
        tstcc_backbone = prepare_tstcc_backbone(case_row, arg_ns, device)
    else:
        print(f"[ALIGN] {pair_id}: reusing cached candidate evaluations for all {len(candidate_rows)} searched candidates.")

    cache_changed = False
    for cand in candidate_rows:
        arch_text = cand["arch"]
        if arch_text in cache and not args.force:
            continue
        set_global_seed(arg_ns.seed)
        eval_out = evaluate_candidate_arch(
            arch_text=arch_text,
            arrays=arrays,
            args=arg_ns,
            device=device,
            tstcc_backbone=tstcc_backbone,
            case_out_dir=case_out_dir / "final_only_cache",
        )
        cache[arch_text] = eval_out
        cache_changed = True
        dump_json(cache_path, cache)

    if cache_changed or not cache_path.exists():
        dump_json(cache_path, cache)

    merged = []
    for cand in candidate_rows:
        eval_out = cache.get(cand["arch"])
        if not eval_out:
            continue
        merged.append(
            {
                **cand,
                "val_auroc": eval_out.get("val_auroc"),
                "val_auprc": eval_out.get("val_auprc"),
                "val_f1_best": eval_out.get("val_f1_best"),
            }
        )

    merged = [row for row in merged if row.get("val_auroc") is not None]
    if len(merged) < 2:
        return None, merged

    merged.sort(key=lambda r: (r["upper_obj_min"], r["arch"]))
    for idx, row in enumerate(merged, start=1):
        row["upper_rank"] = idx

    by_val = sorted(
        merged,
        key=lambda r: (
            -(r["val_auroc"] if r["val_auroc"] is not None else float("-inf")),
            -(r["val_auprc"] if r["val_auprc"] is not None else float("-inf")),
            r["upper_obj_min"],
            r["arch"],
        ),
    )
    val_rank_lookup = {row["arch"]: idx for idx, row in enumerate(by_val, start=1)}
    for row in merged:
        row["val_rank"] = val_rank_lookup[row["arch"]]

    best_by_upper = merged[0]
    best_by_val = by_val[0]
    topk_hits = {}
    upper_top_archs = [row["arch"] for row in merged]
    for k in topk_values:
        cutoff = min(k, len(upper_top_archs))
        topk_hits[k] = best_by_val["arch"] in upper_top_archs[:cutoff]

    search_scores = [-row["upper_obj_min"] for row in merged]
    val_aurocs = [row["val_auroc"] for row in merged]
    val_auprcs = [row["val_auprc"] for row in merged]

    case_summary = {
        "pair_id": pair_id,
        "n_unique_candidates": len(merged),
        "best_upper_arch": best_by_upper["arch"],
        "best_upper_val_auroc": best_by_upper["val_auroc"],
        "best_upper_val_auprc": best_by_upper["val_auprc"],
        "best_upper_val_rank": best_by_upper["val_rank"],
        "best_val_arch": best_by_val["arch"],
        "best_val_auroc": best_by_val["val_auroc"],
        "best_val_auprc": best_by_val["val_auprc"],
        "best_val_upper_rank": best_by_val["upper_rank"],
        "pearson_searchscore_vs_val_auroc": pearson_corr(search_scores, val_aurocs),
        "spearman_searchscore_vs_val_auroc": spearman_corr(search_scores, val_aurocs),
        "pearson_searchscore_vs_val_auprc": pearson_corr(search_scores, val_auprcs),
        "spearman_searchscore_vs_val_auprc": spearman_corr(search_scores, val_auprcs),
        "topk_hits": {str(k): bool(v) for k, v in topk_hits.items()},
        "pad_rank": case_row.get("global_pad_rank"),
        "pad_value": (case_row.get("candidate_pair_shift_precheck") or {}).get("pad_value"),
    }
    return case_summary, merged


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if math.isnan(value):
                    value = ""
                else:
                    value = f"{value:.6f}"
            vals.append(json.dumps(value) if isinstance(value, str) else str(value))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def build_report(summary, case_summaries, candidate_rows_by_case, topk_values):
    lines = []
    lines.append("# NAS Search Alignment Report")
    lines.append("")
    lines.append(f"- Benchmark root: `{summary['benchmark_root']}`")
    lines.append(f"- Dataset: `{summary.get('dataset_name', 'unknown')}`")
    lines.append(f"- Manifest: `{summary.get('manifest', '')}`")
    lines.append(f"- Cases analyzed: `{len(case_summaries)}`")
    lines.append(f"- Top-k evaluated: `{', '.join(str(k) for k in topk_values)}`")
    lines.append("")

    if case_summaries:
        pooled_rows = [row for rows in candidate_rows_by_case.values() for row in rows]
        pooled_search = [-row["upper_obj_min"] for row in pooled_rows]
        pooled_auroc = [row["val_auroc"] for row in pooled_rows]
        pooled_auprc = [row["val_auprc"] for row in pooled_rows]
        lines.append("## Overall")
        lines.append("")
        lines.append(f"- Pooled Pearson(search_score, val AUROC): `{format_metric(pearson_corr(pooled_search, pooled_auroc))}`")
        lines.append(f"- Pooled Spearman(search_score, val AUROC): `{format_metric(spearman_corr(pooled_search, pooled_auroc))}`")
        lines.append(f"- Pooled Pearson(search_score, val AUPRC): `{format_metric(pearson_corr(pooled_search, pooled_auprc))}`")
        lines.append(f"- Pooled Spearman(search_score, val AUPRC): `{format_metric(spearman_corr(pooled_search, pooled_auprc))}`")
        for k in topk_values:
            hit_rate = np.mean([1.0 if cs["topk_hits"].get(str(k)) else 0.0 for cs in case_summaries])
            lines.append(f"- Top-{k} hit rate: `{hit_rate:.4f}`")
        lines.append("")

        lines.append("## Per Case")
        lines.append("")
        lines.append("| Pair | PAD rank | PAD | n cand | Pearson AUROC | Spearman AUROC | Best upper val rank | Best val upper rank | Top-1 | Top-3 | Top-5 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cs in case_summaries:
            lines.append(
                "| {pair_id} | {pad_rank} | {pad_value:.4f} | {n_unique_candidates} | {pear} | {spear} | {best_upper_val_rank} | {best_val_upper_rank} | {t1} | {t3} | {t5} |".format(
                    pair_id=cs["pair_id"],
                    pad_rank=cs["pad_rank"],
                    pad_value=float(cs["pad_value"] or 0.0),
                    n_unique_candidates=cs["n_unique_candidates"],
                    pear=format_metric(cs["pearson_searchscore_vs_val_auroc"]),
                    spear=format_metric(cs["spearman_searchscore_vs_val_auroc"]),
                    best_upper_val_rank=cs["best_upper_val_rank"],
                    best_val_upper_rank=cs["best_val_upper_rank"],
                    t1="Y" if cs["topk_hits"].get("1") else "N",
                    t3="Y" if cs["topk_hits"].get("3") else "N",
                    t5="Y" if cs["topk_hits"].get("5") else "N",
                )
            )
        lines.append("")

        lines.append("## Candidate Tables")
        lines.append("")
        for cs in case_summaries:
            rows = candidate_rows_by_case[cs["pair_id"]]
            lines.append(f"### {cs['pair_id']}")
            lines.append("")
            lines.append("| upper rank | val rank | upper_obj | val AUROC | val AUPRC | family | occurrences |")
            lines.append("|---:|---:|---:|---:|---:|---|---:|")
            for row in rows:
                lines.append(
                    f"| {row['upper_rank']} | {row['val_rank']} | {row['upper_obj_min']:.4f} | "
                    f"{row['val_auroc']:.4f} | {row['val_auprc']:.4f} | {row['families']} | {row['occurrences']} |"
                )
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main():
    args = parse_args()
    benchmark_root = (REPO_ROOT / args.benchmark_root).resolve()
    summary_path = benchmark_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    benchmark_summary = load_json(summary_path)
    benchmark_summary["benchmark_root"] = str(benchmark_root)
    manifest_path = (REPO_ROOT / benchmark_summary["manifest"]).resolve()
    manifest_rows = load_json(manifest_path)
    manifest_map = {pair_id_from_manifest_row(row): row for row in manifest_rows}

    output_root = (REPO_ROOT / args.output_root).resolve() if args.output_root else (benchmark_root / "nas_search_alignment").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested_pairs = {p.strip() for p in args.pair_ids.split(",") if p.strip()}
    topk_values = [int(x) for x in args.topk.split(",") if x.strip()]

    case_summaries = []
    candidate_rows_all = []
    candidate_rows_by_case = {}

    for row in benchmark_summary.get("rows") or []:
        pair_id = row["pair_id"]
        if requested_pairs and pair_id not in requested_pairs:
            continue
        case_row = manifest_map.get(pair_id)
        if case_row is None:
            print(f"[ALIGN] Skipping {pair_id}: not found in manifest.")
            continue
        case_summary, candidate_rows = evaluate_case(
            case_row=case_row,
            summary=benchmark_summary,
            benchmark_root=benchmark_root,
            output_root=output_root,
            args=args,
            topk_values=topk_values,
        )
        if case_summary is None:
            print(f"[ALIGN] Skipping {pair_id}: insufficient candidate data.")
            continue
        case_summaries.append(case_summary)
        candidate_rows_by_case[pair_id] = candidate_rows
        for cand in candidate_rows:
            candidate_rows_all.append(
                {
                    "pair_id": pair_id,
                    "pad_rank": case_summary["pad_rank"],
                    "pad_value": case_summary["pad_value"],
                    **cand,
                }
            )

    case_summaries.sort(key=lambda r: (r["pad_rank"] if r["pad_rank"] is not None else 10**9, r["pair_id"]))
    candidate_rows_all.sort(key=lambda r: (r["pair_id"], r["upper_rank"], r["arch"]))

    dump_json(output_root / "case_summary.json", case_summaries)
    dump_json(output_root / "candidate_rows.json", candidate_rows_all)
    write_csv(
        output_root / "case_summary.csv",
        case_summaries,
        [
            "pair_id",
            "pad_rank",
            "pad_value",
            "n_unique_candidates",
            "pearson_searchscore_vs_val_auroc",
            "spearman_searchscore_vs_val_auroc",
            "pearson_searchscore_vs_val_auprc",
            "spearman_searchscore_vs_val_auprc",
            "best_upper_val_rank",
            "best_val_upper_rank",
            "best_upper_val_auroc",
            "best_val_auroc",
        ],
    )
    write_csv(
        output_root / "candidate_rows.csv",
        candidate_rows_all,
        [
            "pair_id",
            "pad_rank",
            "pad_value",
            "upper_rank",
            "val_rank",
            "upper_obj_min",
            "val_auroc",
            "val_auprc",
            "val_f1_best",
            "families",
            "occurrences",
            "arch",
        ],
    )
    report_text = build_report(benchmark_summary, case_summaries, candidate_rows_by_case, topk_values)
    (output_root / "REPORT.md").write_text(report_text, encoding="utf-8")
    print(f"[ALIGN] Wrote analysis to {output_root}")


if __name__ == "__main__":
    main()
