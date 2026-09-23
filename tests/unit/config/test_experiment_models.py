"""Tests for config models added in Phase 2."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_sdk.config import (
    BM25RetrievalConfig,
    ExperimentConfig,
    FusionConfig,
    HybridRetrievalConfig,
    OptimizationConfig,
    RagConfig,
)


def test_discriminated_union_resolves_retrieval_variants() -> None:
    config = RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive"},
            "retrieval": {"strategy": "hybrid"},
        }
    )
    assert isinstance(config.retrieval, HybridRetrievalConfig)
    assert config.retrieval.top_k == 5


def test_bm25_retrieval_defaults() -> None:
    config = BM25RetrievalConfig()
    assert config.k1 == 1.5
    assert config.b == 0.75
    assert config.stopwords is True


def test_hybrid_retrieval_nested_params() -> None:
    config = HybridRetrievalConfig(
        bm25={"k1": 1.2},
        fusion={"method": "weighted", "dense_weight": 0.3, "bm25_weight": 0.7},
    )
    assert config.bm25.k1 == 1.2
    assert config.fusion.method == "weighted"
    assert config.fusion.candidate_k == 50


def test_weighted_fusion_rejects_zero_weights() -> None:
    with pytest.raises(ValidationError):
        FusionConfig(method="weighted", dense_weight=0.0, bm25_weight=0.0)


def test_unknown_retrieval_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        RagConfig.model_validate(
            {
                "chunking": {"strategy": "recursive"},
                "retrieval": {"strategy": "magic"},
            }
        )


def test_experiment_config_validation() -> None:
    config = ExperimentConfig(
        dataset="queries.jsonl",
        parameters={"chunking.chunk_size": [256, 512]},
        primary_metric="recall_at_k",
        output_dir="out",
    )
    assert config.k == 10
    assert config.primary_metric == "recall_at_k"
    with pytest.raises(ValidationError):
        ExperimentConfig(dataset="q.jsonl", primary_metric="not_a_metric")


def test_rag_config_with_experiments() -> None:
    config = RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive"},
            "documents": {"path": "./docs"},
            "experiments": {"dataset": "queries.jsonl"},
        }
    )
    assert config.documents is not None
    assert config.documents.path == "./docs"
    assert config.experiments is not None


def test_dump_config_excludes_none() -> None:
    from rag_sdk.config import dump_config

    config = RagConfig.model_validate({"chunking": {"strategy": "recursive"}})
    yaml_text = dump_config(config)
    assert "experiments" not in yaml_text
    assert "documents" not in yaml_text

@pytest.mark.parametrize(
    "metric", ["hit_at_k", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map"]
)
def test_optimization_accepts_every_retrieval_metric(metric: str) -> None:
    config = OptimizationConfig(primary_metric=metric, secondary_metric=metric)
    assert config.primary_metric == metric


def test_experiment_relevance_level_defaults_to_document() -> None:
    assert ExperimentConfig(dataset="q.jsonl").relevance_level == "document"
    with pytest.raises(ValidationError):
        ExperimentConfig(dataset="q.jsonl", relevance_level="paragraph")
