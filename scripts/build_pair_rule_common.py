import csv
import json
import math
from pathlib import Path


RULE_CROSS_ENTITY_HARD_TOPPAD_GLOBAL_TOP3 = "cross_entity_hard_topPAD_global_top3"
RULE_CROSS_ENTITY_LEARNABLE_SHIFT_VAL_RICH = "cross_entity_learnable_shift_val_rich"
RULE_CROSS_ENTITY_HARD_LEARNABLE = "cross_entity_hard_learnable"
RULE_CROSS_ENTITY_HARD_LEARNABLE_QBAND = "cross_entity_hard_learnable_qband"
RULE_CROSS_ENTITY_PAPER_SAFE = "cross_entity_paper_safe"
RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW_QUANTILE = "cross_entity_hard_adaptation_window_quantile"
RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW = "cross_entity_hard_adaptation_window"
RULE_CROSS_ENTITY_HARD_ADAPTATION_WINDOW_STABLE = "cross_entity_hard_adaptation_window_stable"
RULE_CROSS_ENTITY_HARD_LEARNABLE_BALANCED = "cross_entity_hard_learnable_balanced"
RULE_CROSS_ENTITY_SAME_APP_DISTURBED_MODERATE_PAD = "cross_entity_same_app_disturbed_moderate_pad"
RULE_CROSS_ENTITY_SAME_APP_ADAPTATION_WINDOW = "cross_entity_same_app_adaptation_window"
RULE_CROSS_ENTITY_SAME_APP_NARROW_ADAPTATION_WINDOW = "cross_entity_same_app_narrow_adaptation_window"
RULE_CROSS_ENTITY_SAME_APP_STABLE_ADAPTATION_WINDOW = "cross_entity_same_app_stable_adaptation_window"


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(payload), f, indent=2, ensure_ascii=False, allow_nan=False)


def safe_float(value, default=float("nan")):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp01(value):
    value = safe_float(value, default=0.0)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def normalize_linear(value, low, high):
    low = safe_float(low, default=0.0)
    high = safe_float(high, default=1.0)
    value = safe_float(value, default=low)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def normalize_ratio(value, ref):
    value = safe_float(value, default=0.0)
    ref = safe_float(ref, default=1.0)
    if not math.isfinite(value) or not math.isfinite(ref) or ref <= 0:
        return 0.0
    return clamp01(value / ref)


def safe_ratio(numerator, denominator, default=float("nan")):
    numerator = safe_float(numerator, default=default)
    denominator = safe_float(denominator, default=default)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return default
    return float(numerator / denominator)


def finite_quantile(values, quantile):
    finite_values = sorted(
        safe_float(value) for value in values if math.isfinite(safe_float(value))
    )
    if not finite_values:
        return float("nan")
    if len(finite_values) == 1:
        return float(finite_values[0])
    q = clamp01(quantile)
    pos = q * (len(finite_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(finite_values[lo])
    frac = pos - lo
    return float(finite_values[lo] * (1.0 - frac) + finite_values[hi] * frac)


def pair_id(source_entity: str, target_entity: str) -> str:
    return f"{source_entity}__to__{target_entity}"


def load_pad_pair_rankings(rankings_path: Path):
    payload = json.loads(rankings_path.read_text(encoding="utf-8"))
    raw_rows = payload.get("pair_rankings") or []
    rows = []
    for raw in raw_rows:
        source_entity = raw.get("source_entity")
        target_entity = raw.get("target_entity")
        if not source_entity or not target_entity:
            continue
        pad_latent = raw.get("pad_latent") or {}
        rows.append(
            {
                "source_entity": str(source_entity),
                "target_entity": str(target_entity),
                "pair_id": pair_id(str(source_entity), str(target_entity)),
                "pad_value": safe_float(pad_latent.get("pad_value")),
                "pad_domain_acc": safe_float(pad_latent.get("domain_acc")),
                "pad_domain_auc": safe_float(pad_latent.get("domain_auc")),
                "pad_feature_mean_l2": safe_float(pad_latent.get("feature_mean_l2")),
                "n_source_normal_windows": int(raw.get("n_source_normal_windows", 0) or 0),
                "n_target_normal_windows": int(raw.get("n_target_normal_windows", 0) or 0),
                "family": raw.get("family", ""),
                "raw_row": raw,
            }
        )

    rows.sort(
        key=lambda row: (
            safe_float(row.get("pad_value"), default=-1.0),
            safe_float(row.get("pad_domain_acc"), default=-1.0),
            safe_float(row.get("pad_domain_auc"), default=-1.0),
            int(row.get("n_target_normal_windows", 0)),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["global_pad_rank"] = idx
    return payload, rows


def public_rows(rows):
    return [{k: sanitize_json(v) for k, v in row.items() if k != "raw_row"} for row in rows]


def compute_learnable_shift_val_rich_score(
    *,
    pad_value,
    source_vs_pool_domain_auc,
    target_pool_hidden_anomaly_ratio,
    val_count,
    val_anomaly_count,
    max_pool_anom_ratio,
    val_count_ref,
    val_anom_ref,
    min_pad_value=1.0,
    max_pad_value=2.0,
):
    pad_strength = normalize_linear(pad_value, min_pad_value, max_pad_value)
    pool_auc_strength = normalize_linear(source_vs_pool_domain_auc, 0.5, 1.0)
    shift_strength = 0.6 * pad_strength + 0.4 * pool_auc_strength

    pool_cleanliness = 1.0
    max_pool_anom_ratio = safe_float(max_pool_anom_ratio, default=0.0)
    if math.isfinite(max_pool_anom_ratio) and max_pool_anom_ratio > 0:
        pool_cleanliness = clamp01(1.0 - (safe_float(target_pool_hidden_anomaly_ratio, default=max_pool_anom_ratio) / max_pool_anom_ratio))

    val_size_strength = normalize_ratio(val_count, val_count_ref)
    val_anom_strength = normalize_ratio(val_anomaly_count, val_anom_ref)
    val_richness = 0.5 * val_size_strength + 0.5 * val_anom_strength

    score = 0.45 * shift_strength + 0.35 * val_richness + 0.20 * pool_cleanliness
    return {
        "score": float(score),
        "components": {
            "shift_strength": float(shift_strength),
            "pool_cleanliness": float(pool_cleanliness),
            "val_richness": float(val_richness),
            "pad_strength": float(pad_strength),
            "pool_auc_strength": float(pool_auc_strength),
            "val_size_strength": float(val_size_strength),
            "val_anom_strength": float(val_anom_strength),
        },
        "refs": {
            "min_pad_value": safe_float(min_pad_value),
            "max_pad_value": safe_float(max_pad_value),
            "max_pool_anom_ratio": safe_float(max_pool_anom_ratio),
            "val_count_ref": safe_int(val_count_ref),
            "val_anom_ref": safe_int(val_anom_ref),
        },
    }


def compute_hard_learnable_score(
    *,
    pad_value,
    target_pool_hidden_anomaly_ratio,
    val_count,
    val_anomaly_count,
    max_pool_anom_ratio,
    val_count_ref,
    val_anom_ref,
    min_pad_value=1.0,
    max_pad_value=2.0,
):
    pad_strength = normalize_linear(pad_value, min_pad_value, max_pad_value)

    pool_cleanliness = 1.0
    max_pool_anom_ratio = safe_float(max_pool_anom_ratio, default=0.0)
    if math.isfinite(max_pool_anom_ratio) and max_pool_anom_ratio > 0:
        pool_cleanliness = clamp01(
            1.0
            - (
                safe_float(
                    target_pool_hidden_anomaly_ratio,
                    default=max_pool_anom_ratio,
                )
                / max_pool_anom_ratio
            )
        )

    val_size_strength = normalize_ratio(val_count, val_count_ref)
    val_anom_strength = normalize_ratio(val_anomaly_count, val_anom_ref)
    val_richness = 0.5 * val_size_strength + 0.5 * val_anom_strength

    score = 0.45 * pad_strength + 0.35 * val_richness + 0.20 * pool_cleanliness
    return {
        "score": float(score),
        "components": {
            "pad_strength": float(pad_strength),
            "pool_cleanliness": float(pool_cleanliness),
            "val_richness": float(val_richness),
            "val_size_strength": float(val_size_strength),
            "val_anom_strength": float(val_anom_strength),
        },
        "refs": {
            "min_pad_value": safe_float(min_pad_value),
            "max_pad_value": safe_float(max_pad_value),
            "max_pool_anom_ratio": safe_float(max_pool_anom_ratio),
            "val_count_ref": safe_int(val_count_ref),
            "val_anom_ref": safe_int(val_anom_ref),
        },
    }


def build_hard_learnable_methods_text(
    *,
    min_val_count,
    min_val_anom,
    max_pool_anom_ratio,
    precheck_l2_quantile,
):
    quantile_pct = int(round(clamp01(precheck_l2_quantile) * 100))
    pool_pct = round(safe_float(max_pool_anom_ratio, default=0.0) * 100, 1)
    return (
        "We retained only cross-entity pairs instantiated under the hard split protocol. "
        f"Among PAD-ranked candidates, we kept pairs whose validation split contained at least {safe_int(min_val_count)} windows "
        f"and at least {safe_int(min_val_anom)} anomalous windows, and whose unlabeled target-pool hidden anomaly ratio did not exceed {pool_pct}%. "
        f"To avoid extreme, non-learnable shifts, we excluded pairs whose precheck feature-mean L2 exceeded the dataset-specific Q{quantile_pct} cutoff. "
        "Within the retained set, pairs were ranked by global PAD rank, with validation anomaly count, validation size, and target-pool cleanliness used only as tie-breakers."
    )


def compute_hard_learnable_qband_thresholds(
    *,
    rows,
    min_pad_quantile,
    max_precheck_l2_quantile,
    max_pool_anom_ratio_quantile,
    min_val_count_quantile,
    min_val_anom_quantile,
    max_val_anom_ratio_quantile,
    min_pad_floor,
    max_pool_anom_ratio_cap,
    min_val_floor,
    min_val_anom_floor,
    max_val_anom_ratio_cap,
):
    if not rows:
        return {
            "min_pad_value": safe_float(min_pad_floor, default=1.0),
            "max_precheck_l2": float("nan"),
            "max_pool_anom_ratio": safe_float(max_pool_anom_ratio_cap, default=0.10),
            "min_val_count": safe_int(min_val_floor, default=32),
            "min_val_anomaly_count": safe_int(min_val_anom_floor, default=7),
            "max_val_anomaly_ratio": safe_float(max_val_anom_ratio_cap, default=0.40),
        }

    pad_q = finite_quantile([row.get("pad_value") for row in rows], min_pad_quantile)
    precheck_q = finite_quantile([row.get("pad_feature_mean_l2") for row in rows], max_precheck_l2_quantile)
    pool_q = finite_quantile([row.get("target_pool_hidden_anomaly_ratio") for row in rows], max_pool_anom_ratio_quantile)
    val_count_q = finite_quantile([row.get("val_count") for row in rows], min_val_count_quantile)
    val_anom_q = finite_quantile([row.get("val_anomaly_count") for row in rows], min_val_anom_quantile)
    val_anom_ratio_q = finite_quantile([row.get("val_anomaly_ratio") for row in rows], max_val_anom_ratio_quantile)

    min_pad_value = max(
        safe_float(min_pad_floor, default=1.0),
        safe_float(pad_q, default=safe_float(min_pad_floor, default=1.0)),
    )
    max_pool_anom_ratio = safe_float(max_pool_anom_ratio_cap, default=0.10)
    if math.isfinite(safe_float(pool_q)):
        max_pool_anom_ratio = min(max_pool_anom_ratio, safe_float(pool_q))
    min_val_count = max(
        safe_int(min_val_floor, default=32),
        int(math.ceil(safe_float(val_count_q, default=safe_int(min_val_floor, default=32)))),
    )
    min_val_anomaly_count = max(
        safe_int(min_val_anom_floor, default=7),
        int(math.ceil(safe_float(val_anom_q, default=safe_int(min_val_anom_floor, default=7)))),
    )
    max_val_anomaly_ratio = safe_float(max_val_anom_ratio_cap, default=0.40)
    if math.isfinite(safe_float(val_anom_ratio_q)):
        max_val_anomaly_ratio = min(max_val_anomaly_ratio, safe_float(val_anom_ratio_q))

    return {
        "min_pad_value": float(min_pad_value),
        "max_precheck_l2": safe_float(precheck_q, default=float("nan")),
        "max_pool_anom_ratio": float(max_pool_anom_ratio),
        "min_val_count": int(min_val_count),
        "min_val_anomaly_count": int(min_val_anomaly_count),
        "max_val_anomaly_ratio": float(max_val_anomaly_ratio),
    }


def build_hard_learnable_qband_methods_text(
    *,
    min_pad_quantile,
    max_precheck_l2_quantile,
    max_pool_anom_ratio_quantile,
    min_val_count_quantile,
    min_val_anom_quantile,
    max_val_anom_ratio_quantile,
    thresholds,
):
    return (
        "We retained only cross-entity pairs instantiated under the hard split protocol. "
        f"Eligibility thresholds were calibrated per dataset from the buildable candidate-pair distribution: PAD above Q{int(round(clamp01(min_pad_quantile) * 100))}, "
        f"precheck feature-mean L2 at or below Q{int(round(clamp01(max_precheck_l2_quantile) * 100))}, "
        f"target-pool hidden anomaly ratio at or below min(10%, Q{int(round(clamp01(max_pool_anom_ratio_quantile) * 100))}), "
        f"validation size at or above max(32, Q{int(round(clamp01(min_val_count_quantile) * 100))}), "
        f"validation anomaly count at or above max(7, Q{int(round(clamp01(min_val_anom_quantile) * 100))}), "
        f"and validation anomaly ratio at or below min(40%, Q{int(round(clamp01(max_val_anom_ratio_quantile) * 100))}). "
        f"For this dataset, the resulting thresholds were PAD >= {thresholds['min_pad_value']:.4f}, "
        f"precheck L2 <= {thresholds['max_precheck_l2']:.4f}, "
        f"pool anomaly ratio <= {thresholds['max_pool_anom_ratio']:.4f}, "
        f"val count >= {thresholds['min_val_count']}, "
        f"val anomaly count >= {thresholds['min_val_anomaly_count']}, "
        f"and val anomaly ratio <= {thresholds['max_val_anomaly_ratio']:.4f}. "
        "Within the retained set, pairs were ranked by global PAD rank, with validation anomaly count, validation size, and target-pool cleanliness used only as tie-breakers."
    )


def compute_paper_safe_thresholds(
    *,
    rows,
    min_pad_quantile,
    max_pool_anom_ratio_quantile,
    min_val_count_quantile,
    min_val_anom_quantile,
    max_val_anom_ratio_quantile,
    min_pad_floor,
    max_pool_anom_ratio_cap,
    min_val_floor,
    min_val_anom_floor,
    max_val_anom_ratio_cap,
):
    if not rows:
        return {
            "min_pad_value": safe_float(min_pad_floor, default=1.0),
            "max_pool_anom_ratio": safe_float(max_pool_anom_ratio_cap, default=0.10),
            "min_val_count": safe_int(min_val_floor, default=32),
            "min_val_anomaly_count": safe_int(min_val_anom_floor, default=7),
            "max_val_anomaly_ratio": safe_float(max_val_anom_ratio_cap, default=0.40),
        }

    pad_q = finite_quantile([row.get("pad_value") for row in rows], min_pad_quantile)
    pool_q = finite_quantile(
        [row.get("target_pool_hidden_anomaly_ratio") for row in rows],
        max_pool_anom_ratio_quantile,
    )
    val_count_q = finite_quantile([row.get("val_count") for row in rows], min_val_count_quantile)
    val_anom_q = finite_quantile([row.get("val_anomaly_count") for row in rows], min_val_anom_quantile)
    val_anom_ratio_q = finite_quantile(
        [row.get("val_anomaly_ratio") for row in rows],
        max_val_anom_ratio_quantile,
    )

    min_pad_value = max(
        safe_float(min_pad_floor, default=1.0),
        safe_float(pad_q, default=safe_float(min_pad_floor, default=1.0)),
    )
    max_pool_anom_ratio = safe_float(max_pool_anom_ratio_cap, default=0.10)
    if math.isfinite(safe_float(pool_q)):
        max_pool_anom_ratio = min(max_pool_anom_ratio, safe_float(pool_q))
    min_val_count = max(
        safe_int(min_val_floor, default=32),
        int(math.ceil(safe_float(val_count_q, default=safe_int(min_val_floor, default=32)))),
    )
    min_val_anomaly_count = max(
        safe_int(min_val_anom_floor, default=7),
        int(math.ceil(safe_float(val_anom_q, default=safe_int(min_val_anom_floor, default=7)))),
    )
    max_val_anomaly_ratio = safe_float(max_val_anom_ratio_cap, default=0.40)
    if math.isfinite(safe_float(val_anom_ratio_q)):
        max_val_anomaly_ratio = min(max_val_anomaly_ratio, safe_float(val_anom_ratio_q))

    return {
        "min_pad_value": float(min_pad_value),
        "max_pool_anom_ratio": float(max_pool_anom_ratio),
        "min_val_count": int(min_val_count),
        "min_val_anomaly_count": int(min_val_anomaly_count),
        "max_val_anomaly_ratio": float(max_val_anomaly_ratio),
    }


def build_paper_safe_methods_text(
    *,
    preferred_shift_levels,
    min_pad_quantile,
    max_pool_anom_ratio_quantile,
    min_val_count_quantile,
    min_val_anom_quantile,
    max_val_anom_ratio_quantile,
    thresholds,
):
    shift_text = ", then ".join(preferred_shift_levels) if preferred_shift_levels else "hard"
    return (
        "We retained only buildable cross-entity pairs and ranked them in TS-JEPA latent space using PAD. "
        f"For each source-target pair, we instantiated the preferred split order {shift_text}; a later split was used only when an earlier one could not be built. "
        f"Eligibility thresholds were calibrated from the dataset-specific buildable-pair distribution: PAD above Q{int(round(clamp01(min_pad_quantile) * 100))}, "
        f"target-pool hidden anomaly ratio at or below min(10%, Q{int(round(clamp01(max_pool_anom_ratio_quantile) * 100))}), "
        f"validation size at or above max(32, Q{int(round(clamp01(min_val_count_quantile) * 100))}), "
        f"validation anomaly count at or above max(7, Q{int(round(clamp01(min_val_anom_quantile) * 100))}), "
        f"and validation anomaly ratio at or below min(40%, Q{int(round(clamp01(max_val_anom_ratio_quantile) * 100))}). "
        f"For this dataset, the resulting thresholds were PAD >= {thresholds['min_pad_value']:.4f}, "
        f"pool anomaly ratio <= {thresholds['max_pool_anom_ratio']:.4f}, "
        f"val count >= {thresholds['min_val_count']}, "
        f"val anomaly count >= {thresholds['min_val_anomaly_count']}, "
        f"and val anomaly ratio <= {thresholds['max_val_anomaly_ratio']:.4f}. "
        "Within the retained set, pairs were ranked by PAD, with validation anomaly count, validation size, and target-pool cleanliness used only as tie-breakers."
    )


def compute_adaptation_window_quantile_thresholds(
    *,
    rows,
    min_source_pool_l2_quantile,
    max_source_test_l2_quantile,
):
    if not rows:
        return {
            "min_source_vs_target_pool_l2": float("nan"),
            "max_source_vs_test_l2": float("nan"),
        }

    pool_q = finite_quantile(
        [row.get("source_vs_target_pool_feature_mean_l2") for row in rows],
        min_source_pool_l2_quantile,
    )
    test_q = finite_quantile(
        [row.get("source_vs_test_feature_mean_l2") for row in rows],
        max_source_test_l2_quantile,
    )
    return {
        "min_source_vs_target_pool_l2": safe_float(pool_q, default=float("nan")),
        "max_source_vs_test_l2": safe_float(test_q, default=float("nan")),
    }


def build_adaptation_window_quantile_methods_text(
    *,
    min_source_pool_l2_quantile,
    max_source_test_l2_quantile,
    thresholds,
):
    return (
        "We retained only buildable cross-entity pairs instantiated under the hard split protocol. "
        "To focus on learnable distribution shifts, we used a moderate adaptation window defined on split-time feature gaps: "
        f"the source-vs-target-pool feature-mean L2 had to be at or above Q{int(round(clamp01(min_source_pool_l2_quantile) * 100))}, "
        f"while the source-vs-test feature-mean L2 had to remain at or below Q{int(round(clamp01(max_source_test_l2_quantile) * 100))} "
        "of the buildable candidate-pair distribution. "
        f"For this dataset, the resulting thresholds were source-vs-target-pool L2 >= {thresholds['min_source_vs_target_pool_l2']:.4f} "
        f"and source-vs-test L2 <= {thresholds['max_source_vs_test_l2']:.4f}. "
        "Within the retained set, pairs were ranked by PAD, with stronger source-vs-pool separation and lower source-vs-test gap used only as tie-breakers."
    )


def write_table_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: sanitize_json(row.get(col)) for col in columns})


def write_table_markdown(path: Path, rows, columns, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [f"# {title}", "", header, divider]
    for row in rows:
        values = []
        for col in columns:
            value = sanitize_json(row.get(col))
            if value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
