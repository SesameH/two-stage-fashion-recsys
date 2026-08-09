"""Ranking metrics. This module is the project's measuring stick — keep it boring and tested.

MAP@12 follows the Kaggle H&M competition definition:
  - the denominator is min(len(actual), k), not len(actual)
  - a customer with no ground-truth purchases contributes 0.0 and is still counted
    by `mapk`, so callers must filter to buyers themselves before scoring
  - duplicate predictions are only credited once
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = ["apk", "mapk", "ndcg_at_k", "recall_at_k"]


def apk(actual: Iterable, predicted: Sequence, k: int = 12) -> float:
    """Average precision at k for a single customer.

    Args:
        actual: ground-truth item ids (order irrelevant, duplicates ignored).
        predicted: ranked item ids, best first. Only the first k are used.
        k: cutoff.

    Returns:
        AP@k in [0, 1]. Returns 0.0 when `actual` is empty.
    """
    truth = set(actual)
    if not truth:
        return 0.0

    hits = 0
    score = 0.0
    seen: set = set()
    for i, p in enumerate(predicted[:k]):
        if p in seen:
            continue
        seen.add(p)
        if p in truth:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(truth), k)


def mapk(actual: Sequence[Sequence], predicted: Sequence[Sequence], k: int = 12) -> float:
    """Mean average precision at k over customers.

    `actual` and `predicted` must be aligned row-wise: row i of each is the same customer.
    """
    if len(actual) != len(predicted):
        raise ValueError(f"length mismatch: {len(actual)} actual vs {len(predicted)} predicted")
    if not actual:
        return 0.0
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


def recall_at_k(actual: Sequence[Sequence], predicted: Sequence[Sequence], k: int = 100) -> float:
    """Mean per-customer recall@k. This is the retrieval stage's ceiling on MAP@12."""
    if len(actual) != len(predicted):
        raise ValueError(f"length mismatch: {len(actual)} actual vs {len(predicted)} predicted")

    scores = []
    for a, p in zip(actual, predicted):
        truth = set(a)
        if not truth:
            continue
        scores.append(len(truth & set(p[:k])) / len(truth))

    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(actual: Sequence[Sequence], predicted: Sequence[Sequence], k: int = 12) -> float:
    """Mean NDCG@k with binary relevance — the ranking stage's auxiliary metric."""
    if len(actual) != len(predicted):
        raise ValueError(f"length mismatch: {len(actual)} actual vs {len(predicted)} predicted")

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for a, p in zip(actual, predicted):
        truth = set(a)
        if not truth:
            continue
        gains = np.array([1.0 if item in truth else 0.0 for item in p[:k]])
        dcg = float((gains * discounts[: len(gains)]).sum())
        ideal = float(discounts[: min(len(truth), k)].sum())
        scores.append(dcg / ideal)

    return float(np.mean(scores)) if scores else 0.0
