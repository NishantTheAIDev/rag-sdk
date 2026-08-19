"""Tests for configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_sdk.config import (
    FixedTokenChunkerConfig,
    RagConfig,
    RecursiveChunkerConfig,
)


def _config(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "chunking": {"strategy": "recursive", "chunk_size": 512, "overlap": 64}
    }
    data.update(overrides)
    return data


def test_recursive_config_defaults() -> None:
    cfg = RagConfig.model_validate(_config())
    assert isinstance(cfg.chunking, RecursiveChunkerConfig)
    assert cfg.chunking.separators == ["\n\n", "\n", ". ", " "]
    assert cfg.retrieval.top_k == 5
    assert cfg.project.name == "rag-project"


def test_fixed_config_discriminated() -> None:
    cfg = RagConfig.model_validate(
        _config(chunking={"strategy": "fixed", "chunk_size": 256, "overlap": 32})
    )
    assert isinstance(cfg.chunking, FixedTokenChunkerConfig)
    assert cfg.chunking.tokenizer == "whitespace"


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        RagConfig.model_validate(_config(chunking={"strategy": "bogus"}))


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        RagConfig.model_validate(
            _config(chunking={"strategy": "recursive", "chunk_size": 10, "overlap": 10})
        )


def test_extra_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        RagConfig.model_validate(_config(unknown_field="x"))


def test_round_trip_serialization() -> None:
    cfg = RagConfig.model_validate(_config())
    restored = RagConfig.model_validate(cfg.model_dump(mode="json"))
    assert restored == cfg