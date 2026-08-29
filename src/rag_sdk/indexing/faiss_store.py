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
        self._metadata: dict[str, dict] = {}
        # Store embeddings for retrieval by ID (for auto-merging)
        self._embeddings: dict[str, np.ndarray] = {}

    def add(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        metadata: dict[str, dict] | None = None,
    ) -> None:
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
# Store metadata for filtering
        if metadata:
            for _idx, chunk_id in enumerate(ids):
                if chunk_id in metadata:
                    self._metadata[chunk_id] = metadata[chunk_id]

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filters: dict[str, str | int | float | bool | list[str] | list[int]] | None = None,
    ) -> list[tuple[str, float]]:
        if k < 1:
            return []
        if len(self) == 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise ValueError(
                f"query vector must have {self._dimension} dimensions, got {query.shape[1]}"
            )

        # If filters are provided, we need to filter before searching
        # For FAISS, we'll do post-filtering (search more, then filter)
        limit = min(k * 10, len(self)) if filters else min(k, len(self))

        scores, indices = self._index.search(query, limit)
        
        results = []
        for score, i in zip(scores[0], indices[0], strict=True):
            if i == -1:
                continue
            chunk_id = self._ids[i]
            
            # Apply metadata filters
            if filters:
                chunk_meta = self._metadata.get(chunk_id, {})
                if not self._matches_filters(chunk_meta, filters):
                    continue
            
            results.append((chunk_id, float(score)))
            if len(results) >= k:
                break
        
        return results

    def _matches_filters(
        self,
        metadata: dict,
        filters: dict[str, str | int | float | bool | list[str] | list[int]],
    ) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            meta_value = metadata[key]
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            elif meta_value != value:
                return False
        return True

    def __len__(self) -> int:
        return self._index.ntotal

    def get_embedding(self, chunk_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a chunk ID."""
        return self._embeddings.get(chunk_id)