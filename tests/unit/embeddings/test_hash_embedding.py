"""Tests for the hash embedding provider and embedding factory."""

from __future__ import annotations

import numpy as np

from rag_sdk.config import EmbeddingConfig
from rag_sdk.embeddings import HashEmbeddingProvider, build_embedding_provider


def test_hash_provider_dimension_and_normalization() -> None:
    provider = HashEmbeddingProvider(dimension=64)
    vectors = provider.embed(["cats are mammals", "banks lend money"])

    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_hash_provider_is_deterministic() -> None:
    provider = HashEmbeddingProvider()
    first = provider.embed(["same text here"])
    second = provider.embed(["same text here"])
    assert np.array_equal(first, second)


def test_build_embedding_provider_hash_default() -> None:
    provider = build_embedding_provider(EmbeddingConfig(provider="hash"))
    assert isinstance(provider, HashEmbeddingProvider)
    assert provider.dimension == 128


def test_build_embedding_provider_custom_dimension() -> None:
    provider = build_embedding_provider(
        EmbeddingConfig(provider="hash", dimension=32)
    )
    assert provider.dimension == 32


def test_build_embedding_provider_unknown() -> None:
    import pytest

    from rag_sdk.embeddings import build_embedding_provider

    with pytest.raises(KeyError):
        build_embedding_provider(EmbeddingConfig(provider="unknown"))