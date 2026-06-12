import numpy as np
from sklearn.metrics import roc_curve, auc

def roc_stats(scores: np.ndarray, y_true: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return {"fpr": fpr, "tpr": tpr, "thr": thr, "auc": float(auc(fpr, tpr))}

def confusion_at_threshold(scores: np.ndarray, y_true: np.ndarray, thr: float):
    y_pred = (scores > thr).astype(int)
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"cm": np.array([[TN, FP], [FN, TP]], dtype=int)}

def threshold_by_quantile(scores: np.ndarray, q: float) -> float:
    return float(np.quantile(scores, q))

def best_f1_threshold(scores: np.ndarray, y_true: np.ndarray):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=int)
    thr_list = np.unique(scores)
    best_thr, best_f1 = float(thr_list[0]), -1.0
    for thr in thr_list:
        pred = (scores > thr).astype(int)
        TP = ((pred == 1) & (y == 1)).sum()
        FP = ((pred == 1) & (y == 0)).sum()
        FN = ((pred == 0) & (y == 1)).sum()
        prec = TP / (TP + FP + 1e-12)
        rec = TP / (TP + FN + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1
