from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _decision_matrix(model, x: np.ndarray) -> np.ndarray:
    logits = model.decision_function(x)
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    return logits


class ViP:
    """Visual Intervention Probing via single-stack ensemble.

    Stacks a linear probe on clean hidden states with a causal delta-only model;
    a meta-learner is trained on their concatenated decision logits (validation split).
    """

    def __init__(self):
        self.linear_probe = LogisticRegression(
            penalty="l2", solver="saga", max_iter=1500, random_state=42, n_jobs=1
        )
        self.causal = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42),
        )
        self.meta = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
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

        self.linear_probe.fit(hs_train, labels_train)
        self.causal.fit(delta_train, labels_train)

        stack_val = np.concatenate(
            [_decision_matrix(self.linear_probe, hs_val), _decision_matrix(self.causal, delta_val)],
            axis=1,
        )
        self.meta.fit(stack_val, labels_val)
        return self

    def _stack(self, delta, hs) -> np.ndarray:
        delta = np.asarray(delta, dtype=float).reshape(-1, 1)
        hs = np.asarray(hs, dtype=np.float32)
        return np.concatenate(
            [_decision_matrix(self.linear_probe, hs), _decision_matrix(self.causal, delta)],
            axis=1,
        )

    def hallucination_probability(self, delta, hs) -> np.ndarray:
        return self.meta.predict_proba(self._stack(delta, hs))[:, 1]

    def confidence(self, delta, hs) -> np.ndarray:
        return 1.0 - self.hallucination_probability(delta, hs)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
