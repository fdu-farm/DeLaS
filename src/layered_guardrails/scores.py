"""Shared score orientation for screening and correctness calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-10
UNCERTAINTY_MEASURES = {"AvgEnt", "MaxEnt", "SEnt", "VASE"}
CONFIDENCE_MEASURES = {"DeLaS-Scr", "LinearProbe", "AvgProb", "MaxProb", "RadFlag"}
CALIBRATION_METHODS = ("UC", "TS", "DeLaS-Cal")


def numeric_scores(values):
    values = np.asarray(pd.to_numeric(np.asarray(values), errors="raise"), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    return values


def hallucination_risk(values, measure):
    """Higher values always indicate more hallucination risk; preserve raw ranks."""
    values = numeric_scores(values)
    if measure in UNCERTAINTY_MEASURES:
        return values
    if measure == "SEne":
        return -values
    if measure in CONFIDENCE_MEASURES:
        return 1.0 - values
    raise ValueError(f"Unknown score measure: {measure}")


def as_confidence_pair(validation_values, test_values, measure):
    """Convert both splits using only validation statistics for SEne."""
    validation = numeric_scores(validation_values)
    test = numeric_scores(test_values)
    if measure in UNCERTAINTY_MEASURES:
        validation = np.exp(-np.clip(validation, 0.0, 100.0))
        test = np.exp(-np.clip(test, 0.0, 100.0))
    elif measure == "SEne":
        if validation.size == 0:
            raise ValueError("SEne requires validation scores")
        scale = max(float(np.median(validation)), EPS)
        validation = 1.0 - np.exp(-np.maximum(validation, 0.0) / scale)
        test = 1.0 - np.exp(-np.maximum(test, 0.0) / scale)
    elif measure not in CONFIDENCE_MEASURES:
        raise ValueError(f"Unknown score measure: {measure}")
    return np.clip(validation, EPS, 1.0 - EPS), np.clip(test, EPS, 1.0 - EPS)
