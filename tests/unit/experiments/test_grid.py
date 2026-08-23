"""Tests for experiment parameter grid expansion."""

from __future__ import annotations

import pytest

from rag_sdk.config import HybridRetrievalConfig, RagConfig
from rag_sdk.config.loader import ConfigError
from rag_sdk.experiments import (
    ExperimentParameterWarning,
    apply_override,
    expand_grid,
    expand_grid_with_detail,
)

BASE = RagConfig.model_validate(
    {
        "chunking": {"strategy": "recursive", "chunk_size": 512, "overlap": 64},
        "retrieval": {"strategy": "dense", "top_k": 5},
    }
)


def test_apply_override_nested_value() -> None:
    updated = apply_override(BASE, "chunking.chunk_size", 256)
    assert updated.chunking.chunk_size == 256
    assert updated.chunking.overlap == 64


def test_apply_override_strategy_selects_variant_default() -> None:
    updated = apply_override(BASE, "retrieval.strategy", "hybrid")
    assert isinstance(updated.retrieval, HybridRetrievalConfig)
    assert updated.retrieval.fusion.method == "rrf"


def test_apply_override_unknown_path_raises() -> None:
    with pytest.raises(ConfigError):
        apply_override(BASE, "chunking.nope", 1)
    with pytest.raises(ConfigError):
        apply_override(BASE, "nope.top_k", 1)


def test_apply_override_unknown_strategy_raises() -> None:
    with pytest.raises(ConfigError):
        apply_override(BASE, "retrieval.strategy", "magic")


def test_expand_grid_cartesian_product() -> None:
    variants = expand_grid(
        BASE,
        {
            "chunking.chunk_size": [256, 512],
            "retrieval.strategy": ["dense", "hybrid"],
        },
    )

    assert len(variants) == 4
    chunk_sizes = {variant.chunking.chunk_size for variant in variants}
    strategies = {variant.retrieval.strategy for variant in variants}
    assert chunk_sizes == {256, 512}
    assert strategies == {"dense", "hybrid"}


def test_expand_grid_deep_override_after_strategy() -> None:
    variants = expand_grid(
        BASE,
        {
            "retrieval.strategy": ["hybrid"],
            "retrieval.fusion.method": ["rrf", "weighted"],
        },
    )
    assert len(variants) == 2
    assert {variant.retrieval.fusion.method for variant in variants} == {
        "rrf",
        "weighted",
    }


def test_expand_grid_empty_parameters_single_baseline() -> None:
    variants = expand_grid(BASE, {})
    assert variants == [BASE]


def test_expand_grid_skips_inapplicable_overrides() -> None:
    with pytest.warns(ExperimentParameterWarning):
        variants = expand_grid(
            BASE,
            {
                "retrieval.strategy": ["dense", "hybrid"],
                "retrieval.fusion.method": ["rrf", "weighted"],
            },
        )

    assert len(variants) == 4
    dense_variants = [
        variant for variant in variants if variant.retrieval.strategy == "dense"
    ]
    hybrid_variants = [
        variant for variant in variants if variant.retrieval.strategy == "hybrid"
    ]
    assert len(dense_variants) == 2
    assert len(hybrid_variants) == 2
    assert {v.retrieval.fusion.method for v in hybrid_variants} == {"rrf", "weighted"}


def test_expand_grid_with_detail_records_skipped_parameters() -> None:
    with pytest.warns(ExperimentParameterWarning):
        variants = expand_grid_with_detail(
            BASE,
            {
                "retrieval.strategy": ["dense", "hybrid"],
                "retrieval.fusion.method": ["weighted"],
            },
        )

    assert len(variants) == 2
    dense = next(v for v in variants if v.config.retrieval.strategy == "dense")
    hybrid = next(v for v in variants if v.config.retrieval.strategy == "hybrid")
    assert dense.skipped_parameters == ["retrieval.fusion.method"]
    assert hybrid.skipped_parameters == []
    assert hybrid.config.retrieval.fusion.method == "weighted"


def test_expand_grid_with_detail_no_sweep() -> None:
    variants = expand_grid_with_detail(BASE, {})
    assert len(variants) == 1
    assert variants[0].config == BASE
    assert variants[0].skipped_parameters == []