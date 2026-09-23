"""Retrieval evaluation metrics.

Each metric is computed per query from the ranked list of retrieved ids and
the set of relevant ids. Aggregate helpers average across queries. These
metrics are kept separate from answer-generation evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set


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


Relevance = Set[str] | Mapping[str, int]


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Relevance,
    k: int = 10,
) -> float:
    """Normalized discounted cumulative gain at ``k``.

    ``relevant`` is either a set of relevant ids (binary relevance) or a
    mapping of id to integer grade (graded relevance, gain ``2**grade - 1``).
    """
    if isinstance(relevant, Mapping):
        dcg = sum(
            (2 ** relevant.get(item, 0) - 1) / math.log2(rank + 1)
            for rank, item in enumerate(retrieved[:k], start=1)
        )
        sorted_grades = sorted(relevant.values(), reverse=True)
        ideal = sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(sorted_grades[:k], start=1)
        )
    else:
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(retrieved[:k], start=1)
            if item in relevant
        )
        ideal = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
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


def graded_average_precision(retrieved: Sequence[str], grades: Mapping[str, int]) -> float:
    """Average precision where each rank's precision is grade-weighted.

    Precision at rank ``r`` is ``sum(grades of top r) / (r * max_grade)``,
    averaged over the ranks of relevant (grade > 0) items and divided by the
    total number of relevant items. It lies in ``[0, 1]`` and equals binary
    average precision when all relevant items share the same grade.
    """
    relevant_total = sum(1 for grade in grades.values() if grade > 0)
    if relevant_total == 0:
        return 0.0
    max_grade = max(grades.values())
    cumulative_grade = 0
    precision_sum = 0.0
    seen: set[str] = set()
    for rank, item in enumerate(retrieved, start=1):
        if item in seen:
            continue
        seen.add(item)
        grade = grades.get(item, 0)
        cumulative_grade += grade
        if grade > 0:
            precision_sum += cumulative_grade / (rank * max_grade)
    return precision_sum / relevant_total


def map_graded(results: Sequence[tuple[Sequence[str], Mapping[str, int]]]) -> float:
    """Mean graded average precision across queries."""
    if not results:
        return 0.0
    return sum(graded_average_precision(r, grades) for r, grades in results) / len(results)


def _to_relevant_set(rel: Relevance) -> Set[str]:
    """Ids with positive relevance; grade-0 entries are judged non-relevant."""
    if isinstance(rel, Mapping):
        return {item for item, grade in rel.items() if grade > 0}
    return rel


def evaluate_retrieval(
    results: Sequence[tuple[Sequence[str], Relevance]],
    k: int = 10,
) -> dict[str, float]:
    """Aggregate all retrieval metrics at ``k`` across queries.

    Relevance per query is a set of ids (binary) or an id-to-grade mapping
    (graded). Binary metrics treat grade > 0 as relevant; ``ndcg_at_k`` uses
    the grades. ``map_graded`` is reported only when some query is graded.
    """
    if not results:
        return {
            "hit_at_k": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "map": 0.0,
        }
    count = len(results)
    binary = [(r, _to_relevant_set(rel)) for r, rel in results]
    metrics = {
        "hit_at_k": sum(hit_at_k(r, rel, k) for r, rel in binary) / count,
        "precision_at_k": sum(precision_at_k(r, rel, k) for r, rel in binary) / count,
        "recall_at_k": sum(recall_at_k(r, rel, k) for r, rel in binary) / count,
        "mrr": sum(reciprocal_rank(r, rel) for r, rel in binary) / count,
        "ndcg_at_k": sum(ndcg_at_k(r, rel, k) for r, rel in results) / count,
        "map": sum(average_precision(r, rel) for r, rel in binary) / count,
    }
    if any(isinstance(rel, Mapping) for _, rel in results):
        metrics["map_graded"] = (
            sum(
                graded_average_precision(r, rel)
                if isinstance(rel, Mapping)
                else average_precision(r, rel)
                for r, rel in results
            )
            / count
        )
    return metrics
