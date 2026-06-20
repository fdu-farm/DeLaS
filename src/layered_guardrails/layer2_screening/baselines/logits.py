from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def logits_baselines(logits: np.ndarray, token_ids: np.ndarray | None = None) -> dict[str, float]:
    """Compute AvgProb, MaxProb, AvgEnt and MaxEnt over response tokens."""
    probabilities = softmax(logits)
    if token_ids is None:
        token_ids = np.argmax(logits, axis=-1)
    token_ids = np.asarray(token_ids, dtype=int)
    length = min(len(probabilities), len(token_ids))
    probabilities = probabilities[:length]
    token_ids = token_ids[:length]
    selected = probabilities[np.arange(length), token_ids]
    entropy = -(probabilities * np.log(probabilities.clip(1e-12))).sum(axis=-1)
    return {
        "AvgProb": float(selected.mean()),
        "MaxProb": float(selected.max()),
        "AvgEnt": float(entropy.mean()),
        "MaxEnt": float(entropy.max()),
    }


def sequence_statistics(logits: np.ndarray) -> tuple[list[float], list[float]]:
    probabilities = softmax(logits)
    return (
        np.max(logits, axis=-1).astype(float).tolist(),
        np.max(probabilities, axis=-1).astype(float).tolist(),
    )

