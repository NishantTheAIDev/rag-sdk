"""Dense (embedding-similarity) retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_sdk.core import Chunk
from rag_sdk.embeddings import EmbeddingProvider
from rag_sdk.indexing import VectorStore
from rag_sdk.retrieval.base import RetrievalResult, Retriever


class DenseRetriever(Retriever):
    """Embeds a query and returns the nearest chunks in a vector store.

    Chunks must be registered with :meth:`add_chunks` so results can carry the
    underlying :class:`Chunk`. Any store population path must go through
    ``add_chunks`` to keep the id-to-chunk map in sync.
    """

    def __init__(self, embedding_provider: EmbeddingProvider, store: VectorStore) -> None:
        self._embedding = embedding_provider
        self._store = store
        self._chunks: dict[str, Chunk] = {}

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        vectors = self._embed([chunk.text for chunk in chunks])
        ids = [chunk.id for chunk in chunks]
        self._store.add(ids, vectors)
        self._chunks.update({chunk.id: chunk for chunk in chunks})

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = self._embed([query])[0]
        hits = self._store.search(query_vector, top_k)
        return [
            RetrievalResult(query=query, chunk=self._chunks[chunk_id], score=score)
            for chunk_id, score in hits
        ]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._embedding.embed(texts)