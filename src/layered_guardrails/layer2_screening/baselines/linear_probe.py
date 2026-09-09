"""Independent hidden-state baseline, trained only on the training split."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def fit_linear_probe(hidden, labels):
    hidden = np.asarray(hidden, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Linear Probe training requires both hallucination label classes")
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1500,
        tol=1e-4, random_state=42, n_jobs=1,
    )
    with threadpool_limits(limits=1):
        model.fit(hidden, labels)
    return model


def confidence(model, hidden):
    positive_index = list(model.classes_).index(1)
    return 1.0 - model.predict_proba(np.asarray(hidden, dtype=np.float32))[:, positive_index]
