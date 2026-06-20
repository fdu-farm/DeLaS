from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


EPS = 1e-12


def normalize_grounding(validation, target):
    validation = np.asarray(validation, dtype=float)
    target = np.asarray(target, dtype=float)
    median = np.nanmedian(validation, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    validation = np.where(np.isfinite(validation), validation, median)
    target = np.where(np.isfinite(target), target, median)
    mean = validation.mean(axis=0)
    std = validation.std(axis=0)
    std = np.where(std > EPS, std, 1.0)
    return (validation - mean) / std, (target - mean) / std


def _apply(confidence, grounding, parameters):
    feature_count = grounding.shape[1]
    weights = parameters[:feature_count]
    bias = float(parameters[feature_count])
    temperature = math.exp(float(parameters[feature_count + 1]))
    offset = math.exp(float(parameters[feature_count + 2]))
    gate = np.clip(expit(grounding @ weights + bias), EPS, 1.0 - EPS)
    calibrated = confidence * np.power(gate, 1.0 / temperature) + offset
    return np.clip(calibrated, EPS, 1.0 - EPS)


def fit_vggc(confidence, labels, grounding):
    confidence = np.clip(np.asarray(confidence, dtype=float), EPS, 1.0 - EPS)
    labels = np.asarray(labels, dtype=float)
    grounding = np.asarray(grounding, dtype=float)
    feature_count = grounding.shape[1]

    def objective(parameters):
        calibrated = _apply(confidence, grounding, parameters)
        return -float(
            np.sum(labels * np.log(calibrated) + (1.0 - labels) * np.log1p(-calibrated))
        )

    initial = np.zeros(feature_count + 3)
    initial[-1] = -5.0
    bounds = [(-20, 20)] * (feature_count + 1) + [(-6, 6), (-20, 0)]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    return result.x


def apply_vggc(confidence, grounding, parameters):
    return _apply(
        np.clip(np.asarray(confidence, dtype=float), EPS, 1.0 - EPS),
        np.asarray(grounding, dtype=float),
        np.asarray(parameters, dtype=float),
    )

