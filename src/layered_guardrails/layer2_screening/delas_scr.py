from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def zscore_by_train(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1, 1) for value in (train, val, test)]
    mean = np.nanmean(arrays[0], axis=0, keepdims=True)
    std = np.nanstd(arrays[0], axis=0, keepdims=True) + 1e-8

    def transform(value: np.ndarray) -> np.ndarray:
        value = np.where(np.isfinite(value), value, np.broadcast_to(mean, value.shape))
        result = (value - mean) / std
        return np.where(np.isfinite(result), result, 0.0).astype(np.float32)

    transformed = tuple(transform(value) for value in arrays)
    return (*transformed, float(mean.item()), float(std.item()))


def apply_causal_zscore(value: np.ndarray, mean: float, scale: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1, 1)
    array = np.where(np.isfinite(array), array, mean)
    result = (array - mean) / scale
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


class DeLaSScr:
    """Visual Intervention Probing via single-stack ensemble.

    Stacks a linear probe on clean hidden states with a causal delta-only model;
    a meta-learner is trained on their concatenated decision logits (validation split).
    """

    def __init__(self):
        self.linear_probe = LogisticRegression(
            penalty="l2", C=1.0, solver="saga", max_iter=1500, tol=1e-4,
            random_state=42, n_jobs=1
        )
        self.causal = LogisticRegression(
            penalty="l2", C=1.0, solver="saga", max_iter=1500, tol=1e-4,
            random_state=42, n_jobs=1,
        )
        self.meta = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1500, tol=1e-4,
            random_state=42, n_jobs=1,
        )

    def fit(
        self,
        delta_train, hs_train, labels_train,
        delta_val, hs_val, labels_val,
    ):
        hs_train = np.asarray(hs_train, dtype=np.float32)
        hs_val = np.asarray(hs_val, dtype=np.float32)
        delta_train = np.asarray(delta_train, dtype=float).reshape(-1, 1)
        delta_val = np.asarray(delta_val, dtype=float).reshape(-1, 1)
        labels_train = np.asarray(labels_train, dtype=int)
        labels_val = np.asarray(labels_val, dtype=int)

        causal_train, causal_val, _, self.causal_mean, self.causal_scale = zscore_by_train(
            delta_train, delta_val, delta_val
        )
        with threadpool_limits(limits=1):
            self.linear_probe.fit(hs_train, labels_train)
            self.causal.fit(causal_train, labels_train)
            stack_val = np.column_stack([
                np.asarray(self.linear_probe.decision_function(hs_val), dtype=np.float64),
                np.asarray(self.causal.decision_function(causal_val), dtype=np.float64),
            ])
            self.meta.fit(stack_val, labels_val)
        return self

    def _stack(self, delta, hs) -> np.ndarray:
        delta = apply_causal_zscore(delta, self.causal_mean, self.causal_scale)
        hs = np.asarray(hs, dtype=np.float32)
        return np.column_stack([
            np.asarray(self.linear_probe.decision_function(hs), dtype=np.float64),
            np.asarray(self.causal.decision_function(delta), dtype=np.float64),
        ])

    def hallucination_probability(self, delta, hs) -> np.ndarray:
        return self.meta.predict_proba(self._stack(delta, hs))[:, 1]

    def confidence(self, delta, hs) -> np.ndarray:
        return 1.0 - self.hallucination_probability(delta, hs)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
