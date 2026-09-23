"""Clustering and regression evaluation metrics.

True simulated labels may be used here ONLY for final metrics and plots,
never inside the fitting loop.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def aligned_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Accuracy after optimal one-to-one label matching (Hungarian algorithm)."""
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)
    n_true = int(true_labels.max()) + 1
    n_pred = int(pred_labels.max()) + 1
    n = max(n_true, n_pred)
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels, strict=False):
        confusion[t, p] += 1
    row, col = linear_sum_assignment(-confusion)
    return float(confusion[row, col].sum() / len(true_labels))


def clustering_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    return {
        "ARI": float(adjusted_rand_score(true_labels, pred_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "aligned_accuracy": aligned_accuracy(true_labels, pred_labels),
    }


def nmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized mean squared error: MSE / Var(y_true)."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true))
    return mse / max(var, eps)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "NMSE": nmse(y_true, y_pred),
    }
