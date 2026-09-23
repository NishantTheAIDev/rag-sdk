"""Tests for graded-relevance retrieval metrics."""

from __future__ import annotations

import pytest

from rag_sdk.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    graded_average_precision,
    map_graded,
    ndcg_at_k,
)


def test_ndcg_accepts_frozenset() -> None:
    assert ndcg_at_k(["a"], frozenset({"a"}), 10) == pytest.approx(1.0)


def test_grade_zero_items_are_not_relevant() -> None:
    metrics = evaluate_retrieval([(["x"], {"a": 0, "x": 0})], k=10)
    assert metrics["hit_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0
    assert metrics["map"] == 0.0
    assert metrics["map_graded"] == 0.0


def test_all_graded_map_is_not_zero() -> None:
    metrics = evaluate_retrieval([(["a", "b"], {"a": 3, "b": 1})], k=10)
    assert metrics["map"] == pytest.approx(1.0)
    assert 0.0 < metrics["map_graded"] <= 1.0


def test_mixed_binary_and_graded_map() -> None:
    metrics = evaluate_retrieval([(["a"], {"a"}), (["a"], {"a": 3})], k=10)
    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["map_graded"] == pytest.approx(1.0)


def test_binary_results_have_no_map_graded_key() -> None:
    assert "map_graded" not in evaluate_retrieval([(["a"], {"a"})], k=10)


def test_graded_ap_equals_binary_ap_for_uniform_grades() -> None:
    assert graded_average_precision(["x", "a", "b"], {"a": 2, "b": 2}) == pytest.approx(
        (1 / 2 + 2 / 3) / 2
    )


def test_graded_ap_rewards_higher_grades_first() -> None:
    grades = {"a": 3, "b": 1}
    assert graded_average_precision(["a", "b"], grades) > graded_average_precision(
        ["b", "a"], grades
    )
    assert map_graded([(["a", "b"], grades)]) <= 1.0
