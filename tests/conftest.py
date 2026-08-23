"""Shared test fixtures."""

from __future__ import annotations

import pytest

from rag_sdk.embeddings import HashEmbeddingProvider

DIMENSION = 128


@pytest.fixture
def hash_embedding() -> HashEmbeddingProvider:
    return HashEmbeddingProvider()