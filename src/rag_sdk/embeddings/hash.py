"""Deterministic, key-free embedding provider for offline development and tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from rag_sdk.embeddings.base import EmbeddingProvider

DEFAULT_DIMENSION = 128


class HashEmbeddingProvider(EmbeddingProvider):
    """Embeds text by seeding a pseudo-random vector per token.

    Each token contributes a seeded unit vector so embeddings are
    deterministic and require no external API. Vectors are L2-normalized so
    inner-product search equals cosine similarity.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.split():
                seed = int.from_bytes(
                    hashlib.sha256(token.encode()).digest()[:4], "big"
                )
                rng = np.random.default_rng(seed)
                vectors[row] += rng.standard_normal(self._dimension).astype(
                    np.float32
                )
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors

    @property
    def dimension(self) -> int:
        return self._dimension