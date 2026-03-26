from __future__ import annotations

import numpy as np
from .omni_spot import SPOT


def calc_point2point(predict, actual):
    predict = np.asarray(predict).astype(int)
    actual = np.asarray(actual).astype(int)
    tp = np.sum(predict * actual)
    tn = np.sum((1 - predict) * (1 - actual))
    fp = np.sum(predict * (1 - actual))
    fn = np.sum((1 - predict) * actual)
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return float(f1), float(precision), float(recall), int(tp), int(tn), int(fp), int(fn)


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    OmniAnomaly-style point adjustment.

    The input `score` follows the repo's semantics:
    lower score => more anomalous.
    """
    score = np.asarray(score)
    label = np.asarray(label)
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")

    if pred is None:
        predict = score < threshold
    else:
        predict = np.asarray(pred).astype(bool)

    actual = label > 0.1
    latency = 0
    anomaly_state = False
    anomaly_count = 0

    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                if not predict[j]:
                    predict[j] = True
                    latency += 1
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True

    if calc_latency:
        return predict.astype(int), float(latency / (anomaly_count + 1e-4))
    return predict.astype(int)


def calc_seq(score, label, threshold, calc_latency=False):
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
        out = list(calc_point2point(predict, label))
        out.append(float(latency))
        return out
    predict = adjust_predicts(score, label, threshold, calc_latency=False)
    return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=False):
    """
    OmniAnomaly-style best-F1 threshold search.
    """
    if step_num is None or end is None:
        end = start
        step_num = 1

    search_step = int(step_num)
    search_range = end - start
    search_lower_bound = start
    threshold = search_lower_bound
    best = (-1.0, -1.0, -1.0)
    best_t = 0.0
    best_full = None

    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > best[0]:
            best_t = threshold
            best = tuple(target[:3])
            best_full = target
        if verbose and i % max(1, display_freq) == 0:
            print("cur thr:", threshold, target, best, best_t)

    if best_full is None:
        best_full = calc_seq(score, label, best_t, calc_latency=True)

    return {
        "f1": float(best_full[0]),
        "precision": float(best_full[1]),
        "recall": float(best_full[2]),
        "tp": int(best_full[3]),
        "tn": int(best_full[4]),
        "fp": int(best_full[5]),
        "fn": int(best_full[6]),
        "latency": float(best_full[7]),
        "threshold": float(best_t),
    }


def pot_eval(init_score, score, label, q=1e-3, level=0.98):
    """
    Inputs follow Omni evaluator semantics:
    lower score => more anomalous.
    """
    init_score = np.asarray(init_score).astype(float)
    score = np.asarray(score).astype(float)
    label = np.asarray(label).astype(int)

    spot = SPOT(q=q)
    spot.fit(init_score, score)
    spot.initialize(level=level, min_extrema=True)
    ret = spot.run(with_alarm=True, dynamic=False)
    pot_th = float(-np.mean(ret["thresholds"]))
    pred, latency = adjust_predicts(score, label, threshold=pot_th, calc_latency=True)
    p2p = calc_point2point(pred, label)
    return {
        "pot-f1": float(p2p[0]),
        "pot-precision": float(p2p[1]),
        "pot-recall": float(p2p[2]),
        "pot-TP": int(p2p[3]),
        "pot-TN": int(p2p[4]),
        "pot-FP": int(p2p[5]),
        "pot-FN": int(p2p[6]),
        "pot-threshold": float(pot_th),
        "pot-latency": float(latency),
        "pred": pred,
        "thresholds": np.asarray(ret["thresholds"]).astype(float),
        "init-threshold": float(-ret["init_threshold"]),
    }
