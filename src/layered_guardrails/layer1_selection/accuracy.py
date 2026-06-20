from __future__ import annotations

import numpy as np


def estimate_accuracy(confidence) -> float:
    """Estimate candidate-model accuracy by averaging confidence."""
    values = np.asarray(confidence, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot estimate accuracy from an empty confidence array.")
    return float(values.mean())

