"""FAISS-backed vector store."""

from __future__ import annotations

from collections.abc import Sequence

import faiss
import numpy as np

from rag_sdk.indexing.base import VectorStore


class FaissVectorStore(VectorStore):
    """An in-memory FAISS store using inner-product similarity.

    Input vectors should be normalized so that inner-product scores equal
    cosine similarities (see :class:`rag_sdk.embeddings.EmbeddingProvider`).
    """

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []
        # Store embeddings for retrieval by ID (for auto-merging)
        self._embeddings: dict[str, np.ndarray] = {}

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self._dimension:
            raise ValueError(
                f"vectors must have shape (n, {self._dimension}), got {array.shape}"
            )
        if len(ids) != len(array):
            raise ValueError("ids and vectors must have the same length")
        if len(ids) != len(set(ids)):
            raise ValueError("ids must be unique")
        self._index.add(array)
        self._ids.extend(ids)
        # Store embeddings for get_embedding lookup
        for idx, chunk_id in enumerate(ids):
            self._embeddings[chunk_id] = array[idx].copy()

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k < 1:
            return []
        if len(self) == 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise ValueError(
                f"query vector must have {self._dimension} dimensions, got {query.shape[1]}"
            )
        limit = min(k, len(self))
        scores, indices = self._index.search(query, limit)
        return [
            (self._ids[i], float(score))
            for score, i in zip(scores[0], indices[0], strict=True)
        ]

    def __len__(self) -> int:
        return self._index.ntotal

    def get_embedding(self, chunk_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a chunk ID."""
        return self._embeddings.get(chunk_id)