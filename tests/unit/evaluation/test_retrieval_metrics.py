"""Tests for retrieval evaluation metrics (hand-computed values)."""

from __future__ import annotations

import math

import pytest

from rag_sdk.evaluation import (
    average_precision,
    evaluate_retrieval,
    hit_at_k,
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"c"}
    assert hit_at_k(retrieved, relevant, k=2) == 0
    assert hit_at_k(retrieved, relevant, k=3) == 1


def test_precision_at_k() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=3) == 2 / 3


def test_recall_at_k() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "x"}
    assert recall_at_k(retrieved, relevant, k=1) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_ndcg_at_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(retrieved, relevant, k=4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(retrieved, set(), k=4) == 0.0
    assert ndcg_at_k([], {"a"}, k=4) == 0.0


def test_average_precision() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "e"}
    expected = (1 / 2 + 2 / 4 + 3 / 5) / 3
    assert average_precision(retrieved, relevant) == pytest.approx(expected)


def test_map_and_evaluate_retrieval() -> None:
    results = [
        (["a", "b", "c"], {"a"}),
        (["x", "y", "z"], {"x", "y"}),
    ]
    metrics = evaluate_retrieval(results, k=3)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["precision_at_k"] == pytest.approx((1 / 3 + 2 / 3) / 2)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 1.0) / 2)
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx(1.0)
    assert map(results) == pytest.approx(1.0)


def test_evaluate_retrieval_empty() -> None:
    metrics = evaluate_retrieval([], k=10)
    assert all(value == 0.0 for value in metrics.values())