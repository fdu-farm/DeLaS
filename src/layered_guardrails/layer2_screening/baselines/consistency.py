from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EntailmentClusterer:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
        self.device = device

    @torch.inference_mode()
    def equivalent(self, first: str, second: str) -> bool:
        def implication(a, b):
            inputs = self.tokenizer(a, b, return_tensors="pt", truncation=True).to(self.device)
            return int(F.softmax(self.model(**inputs).logits, dim=-1).argmax(dim=-1).item())

        forward, backward = implication(first, second), implication(second, first)
        return 0 not in (forward, backward) and (forward, backward) != (1, 1)

    def cluster(self, question: str, responses: list[str]) -> list[int]:
        texts = [f"{question} {response}" for response in responses]
        labels = [-1] * len(texts)
        next_label = 0
        for i, first in enumerate(texts):
            if labels[i] != -1:
                continue
            labels[i] = next_label
            for j in range(i + 1, len(texts)):
                if labels[j] == -1 and self.equivalent(first, texts[j]):
                    labels[j] = next_label
            next_label += 1
        return labels


def _cluster_log_probabilities(probability_sequences, semantic_ids):
    sequence_log_probs = np.array(
        [np.mean(np.log(np.clip(values, 1e-12, 1.0))) for values in probability_sequences],
        dtype=float,
    )
    normalizer = np.logaddexp.reduce(sequence_log_probs)
    by_cluster = defaultdict(list)
    for label, value in zip(semantic_ids, sequence_log_probs):
        by_cluster[label].append(value)
    return {
        label: float(np.logaddexp.reduce(values) - normalizer)
        for label, values in by_cluster.items()
    }


def semantic_entropy(probability_sequences, semantic_ids) -> tuple[float, dict[int, float]]:
    cluster_logs = _cluster_log_probabilities(probability_sequences, semantic_ids)
    distribution = {label: math.exp(value) for label, value in cluster_logs.items()}
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in distribution.values())
    return float(entropy), distribution


def vase(clean_probs, noisy_probs, clean_ids, noisy_ids, alpha=1.0) -> float:
    _, clean_dist = semantic_entropy(clean_probs, clean_ids)
    _, noisy_dist = semantic_entropy(noisy_probs, noisy_ids)
    labels = sorted(set(clean_dist) | set(noisy_dist))
    amplified = np.array(
        [clean_dist.get(label, 0.0) + alpha * (clean_dist.get(label, 0.0) - noisy_dist.get(label, 0.0)) for label in labels]
    )
    amplified = np.exp(amplified - amplified.max())
    amplified /= amplified.sum()
    return float(-(amplified * np.log(amplified.clip(1e-12))).sum())


def semantic_energy(logit_sequences, probability_sequences, semantic_ids) -> float:
    clusters = defaultdict(list)
    for index, label in enumerate(semantic_ids):
        clusters[label].append(index)
    energies = []
    for indices in clusters.values():
        mean_logits = [-float(np.mean(logit_sequences[index])) for index in indices]
        energies.append(-sum(mean_logits))
    return float(max(energies))


def consistency_baselines(
    clean_logits,
    noisy_logits,
    base_response,
    clean_responses,
    noisy_responses,
    question,
    clusterer,
    sequence_statistics_fn,
):
    clean_stats = [sequence_statistics_fn(values) for values in clean_logits]
    noisy_stats = [sequence_statistics_fn(values) for values in noisy_logits]
    clean_logit_seq, clean_prob_seq = zip(*clean_stats)
    _, noisy_prob_seq = zip(*noisy_stats)

    all_ids = clusterer.cluster(question, [base_response] + clean_responses + noisy_responses)
    count = len(clean_responses)
    clean_ids = all_ids[1 : count + 1]
    noisy_ids = all_ids[count + 1 :]
    sent, _ = semantic_entropy(clean_prob_seq, clean_ids)
    return {
        "SEnt": sent,
        "SEne": semantic_energy(clean_logit_seq, clean_prob_seq, clean_ids),
        "VASE": vase(clean_prob_seq, noisy_prob_seq, clean_ids, noisy_ids),
        "RadFlag": float(sum(label == all_ids[0] for label in clean_ids) / max(count, 1)),
    }
