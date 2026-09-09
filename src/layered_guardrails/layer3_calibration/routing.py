"""Translate correctness confidence into three-way workflow decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from layered_guardrails.evaluation.metrics import safe_rate
from layered_guardrails.scores import CALIBRATION_METHODS

DEFAULT_POLICIES = {
    "high_coverage": (0.30, 0.50),
    "balanced": (0.40, 0.60),
    "high_precision": (0.50, 0.70),
}


def route(confidence, low, high):
    if not 0 <= low < high <= 1:
        raise ValueError("Routing thresholds must satisfy 0 <= low < high <= 1")
    values = np.asarray(confidence, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Routing requires finite correctness probabilities in [0, 1]")
    return np.where(values <= low, "reject", np.where(values <= high, "review", "accept"))


def routing_tables(frame, measure, policies=None, label_column="correctness_label"):
    if frame.empty:
        raise ValueError("Cannot route an empty sample table")
    policies = DEFAULT_POLICIES if policies is None else policies
    if not policies:
        raise ValueError("At least one routing policy is required")
    samples, summaries = [], []
    for method in CALIBRATION_METHODS:
        for policy, (low, high) in policies.items():
            decisions = route(frame[method], low, high)
            identifiers = [c for c in ("sample_id", "split", label_column) if c in frame]
            result = frame[identifiers].copy()
            result["measure"] = measure
            result["calibration"] = method
            result["policy"] = policy
            result["tau_low"], result["tau_high"] = low, high
            result["confidence"] = frame[method].to_numpy()
            result["route"] = decisions
            samples.append(result)
            row = {
                "measure": measure, "calibration": method, "policy": policy,
                "tau_low": low, "tau_high": high, "num_samples": len(frame),
            }
            for decision in ("reject", "review", "accept"):
                row[f"{decision}_count"] = int(np.sum(decisions == decision))
                row[f"{decision}_fraction"] = float(np.mean(decisions == decision))
            if label_column in frame:
                row["safe_rate"] = safe_rate(frame[label_column], frame[method], high)
            summaries.append(row)
    return pd.concat(samples, ignore_index=True), pd.DataFrame(summaries)
