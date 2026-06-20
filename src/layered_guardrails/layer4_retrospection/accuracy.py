from __future__ import annotations

import numpy as np


def estimate_accuracy(confidence) -> float:
    """Estimate retrospective group accuracy by averaging confidence."""
    values = np.asarray(confidence, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot estimate accuracy from an empty confidence array.")
    return float(values.mean())


def observed_accuracy(labels) -> float:
    """Compute observed accuracy when labels are available for evaluation."""
    values = np.asarray(labels, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute accuracy from an empty label array.")
    return float(values.mean())

