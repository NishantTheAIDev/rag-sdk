"""FAISS-backed vector store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import faiss
import numpy as np

from rag_sdk.core import MetadataFilters, matches_filters
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
        metadata: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        """Add vectors; ids that are already indexed only get their metadata updated.

        Re-adding an existing id (e.g. ingestion followed by a retriever's
        ``add_chunks``) must not create a second FAISS row, otherwise the same
        chunk is returned multiple times per query.
        """
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self._dimension:
            raise ValueError(
                f"vectors must have shape (n, {self._dimension}), got {array.shape}"
            )
        if len(ids) != len(array):
            raise ValueError("ids and vectors must have the same length")
        if len(ids) != len(set(ids)):
            raise ValueError("ids must be unique")
        if metadata:
            for chunk_id in ids:
                if chunk_id in metadata:
                    self._metadata[chunk_id] = dict(metadata[chunk_id])
        new_rows = [idx for idx, chunk_id in enumerate(ids) if chunk_id not in self._embeddings]
        if not new_rows:
            return
        new_array = array[new_rows]
        self._index.add(new_array)
        for row, idx in enumerate(new_rows):
            chunk_id = ids[idx]
            self._ids.append(chunk_id)
            self._embeddings[chunk_id] = new_array[row].copy()

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filters: MetadataFilters | None = None,
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

        # Flat index: with filters, score everything and post-filter so a
        # selective filter can never starve the result list.
        limit = len(self) if filters else min(k, len(self))
        scores, indices = self._index.search(query, limit)

        results: list[tuple[str, float]] = []
        for score, i in zip(scores[0], indices[0], strict=True):
            if i == -1:
                continue
            chunk_id = self._ids[i]
            if filters and not matches_filters(self._metadata.get(chunk_id), filters):
                continue
            results.append((chunk_id, float(score)))
            if len(results) >= k:
                break
        return results

    def __len__(self) -> int:
        return self._index.ntotal

    def get_embedding(self, chunk_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a chunk ID."""
        return self._embeddings.get(chunk_id)