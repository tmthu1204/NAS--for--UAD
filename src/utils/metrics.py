import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

def compute_ap_auroc(y_true, scores):
    """
    y_true: [N] ∈ {0,1}
    scores: [N] anomaly score (cao = bất thường) hoặc prob_anom
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    ap = average_precision_score(y_true, scores)
    try:
        auroc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else 0.5
    except Exception:
        auroc = 0.5
    return ap, auroc

def pot_threshold(train_scores, q=1e-3, level=0.99):
    """
    Very small POT-like thresholding (không phụ thuộc lib): lấy high-quantile trên phần tail.
    - q: tail ratio, chọn top q tail để ước lượng
    - level: quantile trên tail để làm ngưỡng
    Trả về: threshold scalar
    """
    train_scores = np.asarray(train_scores).astype(float)
    if train_scores.size == 0:
        return float("inf")
    k = max(1, int(np.ceil(len(train_scores) * (1.0 - q))))
    tail = np.sort(train_scores)[-k:]  # top tail
    thr = np.quantile(tail, level)
    return float(thr)

def f1_at_threshold(y_true, scores, thr):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    return prec, rec, f1

def best_f1(y_true, scores):
    # oracle (chỉ để tham khảo)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / np.maximum(1e-12, precisions + recalls)
    idx = int(np.nanargmax(f1s))
    return float(np.nanmax(f1s)), float(precisions[idx]), float(recalls[idx]), float(thresholds[min(idx, len(thresholds)-1)])

def segments_from_binary(y):
    """Trả về list các (start, end) 0-index, inclusive, cho các đoạn y==1 liên tiếp."""
    y = np.asarray(y).astype(int)
    segs = []
    i = 0
    n = len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j+1 < n and y[j+1] == 1:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs

def event_f1_and_delay(y_true, y_pred):
    """
    Event-level: một sự kiện GT (segment 1) được coi là TP nếu có >=1 điểm dự đoán trong segment.
    Delay = khoảng cách giữa start của GT và điểm dự đoán đầu tiên nằm trong segment.
    """
    gt_segs = segments_from_binary(y_true)
    pred_segs = segments_from_binary(y_pred)

    # chuyển dự đoán point-wise thành mask để tính delay
    pred_mask = np.asarray(y_pred).astype(int)

    tp, fp = 0, 0
    delays = []
    matched_gt = [False]*len(gt_segs)

    # đếm FP: những pred segments không giao với bất kỳ GT segment
    for (ps, pe) in pred_segs:
        overlap = any(not (pe < gs or ps > ge) for (gs, ge) in gt_segs)
        if not overlap:
            fp += 1

    # TP + Delay
    for gi, (gs, ge) in enumerate(gt_segs):
        # tồn tại một điểm dự đoán trong [gs, ge]
        hit_idxs = np.where(pred_mask[gs:ge+1] == 1)[0]
        if hit_idxs.size > 0:
            tp += 1
            delays.append(hit_idxs[0])  # số bước từ start
            matched_gt[gi] = True

    fn = len(gt_segs) - tp
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    delay_mean = float(np.mean(delays)) if delays else float('inf')
    delay_median = float(np.median(delays)) if delays else float('inf')
    return {'event_precision': prec, 'event_recall': rec, 'event_f1': f1,
            'delay_mean': delay_mean, 'delay_median': delay_median}
