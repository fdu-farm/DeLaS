from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else float("nan")


def ece(labels, confidence, bins=15):
    labels = np.asarray(labels, dtype=float)
    confidence = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - confidence[mask].mean())
    return float(value)


def ace(labels, confidence, bins=15):
    labels = np.asarray(labels, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    order = np.argsort(confidence)
    groups = np.array_split(order, min(bins, len(order)))
    return float(np.mean([abs(labels[group].mean() - confidence[group].mean()) for group in groups]))


def safe_rate(labels, confidence, threshold):
    labels = np.asarray(labels, dtype=float)
    keep = np.asarray(confidence, dtype=float) >= threshold
    return float(labels[keep].mean()) if keep.any() else float("nan")

