"""Hybrid retrieval combining dense and BM25 rankings."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.config import HybridRetrievalConfig
from rag_sdk.core import Chunk
from rag_sdk.retrieval.base import RetrievalResult, Retriever
from rag_sdk.retrieval.fusion import rrf_fuse, weighted_fuse


class HybridRetriever(Retriever):
    """Runs dense and BM25 retrievers and fuses their rankings.

    ``add_chunks`` populates both underlying retrievers. ``search`` gathers
    ``candidate_k`` results from each and returns the fused ``top_k``.
    """

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._config = config or HybridRetrievalConfig()

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._dense.add_chunks(chunks)
        self._lexical.add_chunks(chunks)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k < 1:
            return []
        candidate_k = max(self._config.fusion.candidate_k, top_k)
        dense_results = self._dense.search(query, candidate_k)
        lexical_results = self._lexical.search(query, candidate_k)
        fusion = self._config.fusion
        if fusion.method == "rrf":
            fused = rrf_fuse(dense_results, lexical_results, k=fusion.rrf_k)
        else:
            fused = weighted_fuse(
                dense_results,
                lexical_results,
                dense_weight=fusion.dense_weight,
                lexical_weight=fusion.bm25_weight,
            )
        return fused[:top_k]