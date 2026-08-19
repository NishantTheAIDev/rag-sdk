"""Evaluation framework."""

from __future__ import annotations

from rag_sdk.evaluation.io import load_retrieval_results
from rag_sdk.evaluation.retrieval_metrics import (
    average_precision,
    evaluate_retrieval,
    hit_at_k,
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "average_precision",
    "evaluate_retrieval",
    "hit_at_k",
    "load_retrieval_results",
    "map",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]