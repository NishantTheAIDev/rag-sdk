"""Retrieval evaluation metrics.

Each metric is computed per query from the ranked list of retrieved ids and
the set of relevant ids. Aggregate helpers average across queries. These
metrics are kept separate from answer-generation evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set


def hit_at_k(retrieved: Sequence[str], relevant: Set[str], k: int = 10) -> int:
    """1 if any relevant item appears in the top ``k``, else 0."""
    return int(any(item in relevant for item in retrieved[:k]))


def precision_at_k(
    retrieved: Sequence[str], relevant: Set[str], k: int = 10
) -> float:
    """Fraction of the top ``k`` retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(
    retrieved: Sequence[str], relevant: Set[str], k: int = 10
) -> float:
    """Fraction of relevant items present in the top ``k``."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Reciprocal of the rank of the first relevant item (0 if none)."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Set[str] | dict[str, int],
    k: int = 10,
) -> float:
    """Normalized discounted cumulative gain at ``k``.

    Supports both binary relevance (Set[str]) and graded relevance (dict[str, int]).
    For graded relevance, grades should be integers 0-3.
    """
    if isinstance(relevant, set):
        # Binary relevance
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(retrieved[:k], start=1)
            if item in relevant
        )
        ideal = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
        )
    else:
        # Graded relevance
        dcg = sum(
            (2**relevant.get(item, 0) - 1) / math.log2(rank + 1)
            for rank, item in enumerate(retrieved[:k], start=1)
        )
        # Ideal DCG: sort grades descending
        sorted_grades = sorted(relevant.values(), reverse=True)
        ideal = sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(sorted_grades[:k], start=1)
        )
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Average of precision values at each relevant item in the ranking."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def map(results: Sequence[tuple[Sequence[str], Set[str]]]) -> float:
    """Mean average precision across queries."""
    if not results:
        return 0.0
    return sum(average_precision(retrieved, relevant) for retrieved, relevant in results) / len(
        results
    )


def map_graded(
    results: Sequence[tuple[Sequence[str], dict[str, int]]],
) -> float:
    """Mean average precision with graded relevance.

    Uses the standard MAP formula but with graded relevance:
    precision at rank r = (sum of grades up to r) / r
    """
    if not results:
        return 0.0
    aps = []
    for retrieved, grades in results:
        if not grades:
            aps.append(0.0)
            continue
        # Sort grades by retrieval order
        cumulative_grade = 0
        precision_sum = 0.0
        relevant_count = 0
        for rank, item in enumerate(retrieved, start=1):
            grade = grades.get(item, 0)
            if grade > 0:
                relevant_count += 1
                cumulative_grade += grade
                precision_sum += cumulative_grade / rank
        aps.append(precision_sum / relevant_count if relevant_count > 0 else 0.0)
    return sum(aps) / len(aps)


def _to_relevant_set(rel: Set[str] | dict[str, int]) -> Set[str]:
    """Convert relevance to a set of relevant item IDs."""
    if isinstance(rel, set):
        return rel
    return set(rel.keys())


def evaluate_retrieval(
    results: Sequence[tuple[Sequence[str], Set[str] | dict[str, int]]],
    k: int = 10,
) -> dict[str, float]:
    """Aggregate all retrieval metrics at ``k`` across queries.

    Supports both binary relevance (Set[str]) and graded relevance (dict[str, int]).
    """
    if not results:
        return {
            "hit_at_k": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "map": 0.0,
            "map_graded": 0.0,
        }
    count = len(results)

    # Check if any result has graded relevance
    has_grades = any(isinstance(rel, dict) for _, rel in results)

    base_metrics = {
        "hit_at_k": sum(
            hit_at_k(r, _to_relevant_set(rel), k) for r, rel in results
        ) / count,
        "precision_at_k": sum(
            precision_at_k(r, _to_relevant_set(rel), k) for r, rel in results
        ) / count,
        "recall_at_k": sum(
            recall_at_k(r, _to_relevant_set(rel), k) for r, rel in results
        ) / count,
        "mrr": sum(
            reciprocal_rank(r, _to_relevant_set(rel)) for r, rel in results
        ) / count,
        "ndcg_at_k": sum(ndcg_at_k(r, rel, k) for r, rel in results) / count,
        "map": sum(
            average_precision(r, rel) for r, rel in results if isinstance(rel, set)
        )
        / count
        if any(isinstance(rel, set) for _, rel in results)
        else 0.0,
    }

    if has_grades:
        base_metrics["map_graded"] = sum(
            average_precision(r, set(rel.keys())) for r, rel in results if isinstance(rel, dict)
        ) / sum(1 for _, rel in results if isinstance(rel, dict))

    return base_metrics