"""Evaluation framework for RAG."""

from __future__ import annotations

from rag_sdk.evaluation.answer import (
    AnswerRelevanceEvaluator,
    CitationAccuracyEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    CorrectnessEvaluator,
    EvaluationResult,
    Evaluator,
    FaithfulnessEvaluator,
    LLMAnswerRelevanceEvaluator,
    LLMCorrectnessEvaluator,
    LLMFaithfulnessEvaluator,
    build_evaluators,
)
from rag_sdk.evaluation.io import load_retrieval_results
from rag_sdk.evaluation.pipeline import EvaluationPipeline, evaluate_rag
from rag_sdk.evaluation.retrieval_metrics import (
    average_precision,
    evaluate_retrieval,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_sdk.evaluation.retrieval_metrics import (
    map as map_score,
)

__all__ = [
    # Retrieval metrics
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "average_precision",
    "map_score",
    "evaluate_retrieval",
    # IO
    "load_retrieval_results",
    # Answer evaluation
    "Evaluator",
    "EvaluationResult",
    "FaithfulnessEvaluator",
    "AnswerRelevanceEvaluator",
    "ContextPrecisionEvaluator",
    "ContextRecallEvaluator",
    "CorrectnessEvaluator",
    "CitationAccuracyEvaluator",
    "LLMFaithfulnessEvaluator",
    "LLMAnswerRelevanceEvaluator",
    "LLMCorrectnessEvaluator",
    "build_evaluators",
    # Pipeline
    "EvaluationPipeline",
    "evaluate_rag",
]