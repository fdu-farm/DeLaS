from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def logit(probability, eps=1e-12):
    probability = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(probability) - np.log1p(-probability)


def fit_temperature(confidence, labels) -> float:
    scores = logit(confidence)
    labels = np.asarray(labels, dtype=float)

    def objective(value):
        temperature = math.exp(float(value[0]))
        scaled = scores / temperature
        return float(np.sum(np.logaddexp(0.0, scaled) - labels * scaled))

    result = minimize(objective, np.array([0.0]), method="L-BFGS-B")
    return float(math.exp(float(result.x[0])))


def apply_temperature(confidence, temperature):
    return expit(logit(confidence) / temperature)

