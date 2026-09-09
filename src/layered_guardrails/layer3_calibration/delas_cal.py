from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-10
WEIGHT_BOUND = 50.0
BIAS_BOUND = 50.0
LOG_T_MIN, LOG_T_MAX = -20.0, 20.0
LOG_C_MIN, LOG_C_MAX = -30.0, 0.0


def normalize_grounding(validation, target):
    """Scale VAS, VAC and JN using validation medians and interquartile ranges."""
    validation = np.asarray(validation, dtype=float)
    target = np.asarray(target, dtype=float)
    if np.isnan(validation).any() or np.isnan(target).any():
        raise ValueError("Grounding features contain missing data")
    center = np.median(validation, axis=0)
    q25, q75 = np.percentile(validation, [25, 75], axis=0)
    scale = np.where(q75 - q25 > EPS, q75 - q25, 1.0)
    return (validation - center) / scale, (target - center) / scale


def _predict(base_score, cues, theta, with_jacobian=False):
    """Exact six-parameter DeLaS-Cal map."""
    base_score = np.clip(np.asarray(base_score, dtype=float), EPS, 1.0 - EPS)
    z = cues @ theta[:3] + theta[3]
    grounding = np.clip(expit(z), EPS, 1.0 - EPS)
    inverse_temperature = math.exp(-float(theta[4]))
    log_grounding = np.log(grounding)
    gain = np.exp(np.clip(inverse_temperature * log_grounding, -50.0, 0.0))
    offset = math.exp(float(theta[5]))
    raw = base_score * gain + offset
    probability = np.clip(raw, EPS, 1.0 - EPS)
    if not with_jacobian:
        return probability, raw
    coefficient = base_score * gain * inverse_temperature * (1.0 - grounding)
    jacobian = np.empty((len(base_score), 6), dtype=float)
    jacobian[:, :3] = coefficient[:, None] * cues
    jacobian[:, 3] = coefficient
    jacobian[:, 4] = (
        -base_score * gain * inverse_temperature * log_grounding
    )
    jacobian[:, 5] = offset
    inside = (raw > EPS) & (raw < 1.0 - EPS)
    return probability, jacobian * inside[:, None], raw


def _fit_bce(base_score, labels, cues, maxiter):
    """Validation-only DeLaS-Cal BCE fit with deterministic starts."""
    def objective(theta):
        probability, jacobian, _ = _predict(base_score, cues, theta, True)
        loss = -float(np.mean(
            labels * np.log(probability)
            + (1.0 - labels) * np.log1p(-probability)
        ))
        gradient_probability = (
            (probability - labels)
            / (probability * (1.0 - probability) * len(labels))
        )
        gradient = jacobian.T @ gradient_probability
        return loss, gradient

    identity = np.array([0, 0, 0, 0, 8.0, LOG_C_MIN], dtype=float)
    standard = np.array([0, 0, 0, 0, 0, -5.0], dtype=float)
    prevalence_delta = max(float(np.mean(labels) - np.mean(base_score)), math.exp(LOG_C_MIN))
    prevalence_start = standard.copy()
    prevalence_start[5] = np.clip(math.log(prevalence_delta), LOG_C_MIN, LOG_C_MAX)
    starts = [
        identity,
        standard,
        standard + np.array([0, 0, 0, 2.0, 1.0, -3.0], dtype=float),
        standard + np.array([0, 0, 0, -2.0, 1.0, -3.0], dtype=float),
        prevalence_start,
    ]
    bounds = (
        [(-WEIGHT_BOUND, WEIGHT_BOUND)] * 3
        + [
            (-BIAS_BOUND, BIAS_BOUND),
            (LOG_T_MIN, LOG_T_MAX),
            (LOG_C_MIN, LOG_C_MAX),
        ]
    )
    results = []
    for start in starts:
        result = minimize(
            lambda value: objective(value)[0], start,
            jac=lambda value: objective(value)[1], method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30},
        )
        if np.isfinite(result.fun):
            results.append(result)
    if not results:
        raise RuntimeError("No finite six-parameter DeLaS-Cal fit")
    successful = [result for result in results if result.success]
    return min(successful or results, key=lambda item: float(item.fun))


def fit_delas_cal(confidence, labels, grounding, maxiter=500):
    return _fit_bce(
        np.asarray(confidence, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(grounding, dtype=float),
        maxiter,
    ).x


def apply_delas_cal(confidence, grounding, parameters):
    return _predict(
        np.asarray(confidence, dtype=float),
        np.asarray(grounding, dtype=float),
        np.asarray(parameters, dtype=float),
    )[0]
