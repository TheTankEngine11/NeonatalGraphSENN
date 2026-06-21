"""Shared metric helpers for validation and sweep summaries."""

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


EPS = 1e-12


def safe_float(value, default=np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def safe_auroc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan


def safe_auprc(y_true, y_prob) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return np.nan


def precision_gain(y_true, y_prob, threshold: float) -> float:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pi = np.mean(y_true)
    precision = tp / (tp + fp + EPS)
    return (precision - pi) / ((1.0 - pi) * precision + EPS)


def recall_gain(y_true, y_prob, threshold: float) -> float:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pi = np.mean(y_true)
    recall = tp / (tp + fn + EPS)
    return (recall - pi) / ((1.0 - pi) * recall + EPS)


def auprg(y_true, y_prob) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    pi = np.mean(y_true)

    precision_gain_values = (precision - pi) / ((1.0 - pi) * np.clip(precision, EPS, None))
    recall_gain_values = (recall - pi) / ((1.0 - pi) * np.clip(recall, EPS, None))

    mask = np.isfinite(precision_gain_values) & np.isfinite(recall_gain_values)
    precision_gain_values = precision_gain_values[mask]
    recall_gain_values = recall_gain_values[mask]

    if len(recall_gain_values) == 0:
        return np.nan

    order = np.argsort(recall_gain_values)
    precision_gain_values = precision_gain_values[order]
    recall_gain_values = recall_gain_values[order]

    non_negative = np.where(recall_gain_values >= 0.0)[0]
    if len(non_negative) == 0:
        return 0.0

    first = non_negative[0]
    if first > 0 and recall_gain_values[first] > 0.0:
        x1, y1 = recall_gain_values[first - 1], precision_gain_values[first - 1]
        x2, y2 = recall_gain_values[first], precision_gain_values[first]
        y_at_zero = y1 + (0.0 - x1) * (y2 - y1) / (x2 - x1 + EPS)
        recall_gain_values = np.concatenate(([0.0], recall_gain_values[first:]))
        precision_gain_values = np.concatenate(([y_at_zero], precision_gain_values[first:]))
    else:
        recall_gain_values = recall_gain_values[first:]
        precision_gain_values = precision_gain_values[first:]

    if recall_gain_values[0] > 0.0:
        recall_gain_values = np.concatenate(([0.0], recall_gain_values))
        precision_gain_values = np.concatenate(([precision_gain_values[0]], precision_gain_values))

    if recall_gain_values[-1] < 1.0:
        recall_gain_values = np.concatenate((recall_gain_values, [1.0]))
        precision_gain_values = np.concatenate((precision_gain_values, [0.0]))

    recall_gain_values, unique_idx = np.unique(recall_gain_values, return_index=True)
    precision_gain_values = precision_gain_values[unique_idx]

    return float(auc(recall_gain_values, precision_gain_values))


def youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def kappa_threshold(y_true, y_prob) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    kappas = []
    for t in thresholds:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        kappas.append(cohen_kappa_score(y_true, y_pred))
    return float(thresholds[np.nanargmax(kappas)])


def f2_threshold(y_true, y_prob) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    f2s = []
    for t in thresholds:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        f2s.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    return float(thresholds[np.nanargmax(f2s)])


def compute_binary_metrics(y_true, y_prob, threshold: float) -> Dict[str, float]:
    """Validation metrics with the existing Validation.py output names."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp + EPS),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "AUROC": safe_auroc(y_true, y_prob),
        "AUPRC": safe_auprc(y_true, y_prob),
        "AUPRG": auprg(y_true, y_prob),
        "precision_gain": precision_gain(y_true, y_prob, threshold),
        "recall_gain": recall_gain(y_true, y_prob, threshold),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def summarize_folds(rows):
    df = pd.DataFrame(rows)
    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.agg(["mean", "std", "median", iqr, q1, q3])
    return df, summary


# Backwards-compatible spelling used by older scripts.
F2_threshold = f2_threshold
