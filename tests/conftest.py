"""Shared test fixtures."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from rag_sdk.embeddings import EmbeddingProvider

DIMENSION = 128


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic, key-free embedding provider for tests.

    Each token contributes a seeded pseudo-random unit vector to the text
    embedding. Vectors are L2-normalized so inner-product equals cosine
    similarity.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.split():
                seed = int.from_bytes(
                    hashlib.sha256(token.encode()).digest()[:4], "big"
                )
                rng = np.random.default_rng(seed)
                vector = rng.standard_normal(self._dimension).astype(np.float32)
                vectors[row] += vector
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.fixture
def hash_embedding() -> HashEmbeddingProvider:
    return HashEmbeddingProvider()