from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ViP:
    """Visual Intervention Probing using Delta as its only feature."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        )

    def fit(self, delta, hallucination_labels):
        features = self.scaler.fit_transform(np.asarray(delta, dtype=float).reshape(-1, 1))
        self.classifier.fit(features, np.asarray(hallucination_labels, dtype=int))
        return self

    def hallucination_probability(self, delta):
        features = self.scaler.transform(np.asarray(delta, dtype=float).reshape(-1, 1))
        return self.classifier.predict_proba(features)[:, 1]

    def confidence(self, delta):
        return 1.0 - self.hallucination_probability(delta)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

