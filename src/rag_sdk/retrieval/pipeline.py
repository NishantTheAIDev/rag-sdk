"""Retrieval pipeline with reranking and context enrichment."""

from __future__ import annotations

from rag_sdk.config import RetrievalConfig
from rag_sdk.config.models import RerankerConfig
from rag_sdk.core import Chunk
from rag_sdk.indexing import ChunkStore, DocumentStore, VectorStore
from rag_sdk.reranking.base import RerankerProvider
from rag_sdk.retrieval.auto_merging import AutoMerger
from rag_sdk.retrieval.base import RetrievalResult, Retriever
from rag_sdk.retrieval.parent_child import ParentChildExpander
from rag_sdk.retrieval.sentence_window import SentenceWindowExpander


class RetrievalPipeline:
    """Composable retrieval pipeline: Retriever → Reranker → Enrichment."""

    def __init__(
        self,
        retriever: Retriever,
        vector_store: VectorStore,
        chunk_store: ChunkStore | None,
        document_store: DocumentStore | None,
        reranker: RerankerProvider | None,
        retrieval_config: RetrievalConfig,
        reranker_config: RerankerConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._vector_store = vector_store
        self._chunk_store = chunk_store
        self._document_store = document_store
        self._reranker = reranker
        self._retrieval_config = retrieval_config
        self._reranker_config = reranker_config

        # Build enrichers in order: ParentChild → SentenceWindow → AutoMerge
        self._enrichers: list = []
        if retrieval_config.parent_child.enabled and chunk_store:
            self._enrichers.append(ParentChildExpander(chunk_store))
        if retrieval_config.sentence_window.enabled and document_store:
            self._enrichers.append(
                SentenceWindowExpander(document_store, retrieval_config.sentence_window)
            )
        if retrieval_config.auto_merging.enabled:
            self._enrichers.append(
                AutoMerger(vector_store, retrieval_config.auto_merging)
            )

    def search(self, query: str) -> list[RetrievalResult]:
        """Execute the full retrieval pipeline."""
        # 1. Retrieve candidate_k results
        candidates = self._retriever.search(query, self._retrieval_config.candidate_k)

        # 2. Build source chunk map and attach lineage
        source_chunk_map: dict[str, Chunk] = {}
        for i, r in enumerate(candidates):
            r.source_chunk_id = r.chunk.id
            r.source_chunk_score = r.score
            r.source_chunk_rank = i + 1
            r.child_ids = [r.chunk.id]
            source_chunk_map[r.chunk.id] = r.chunk

        # 3. Rerank (if configured)
        if self._reranker is not None:
            candidates = self._reranker.rerank(
                query, candidates, self._reranker_config.top_k if self._reranker_config else 5
            )
            for i, r in enumerate(candidates):
                r.rerank_rank = i + 1
        else:
            # No reranker: truncate to retrieval.top_k BEFORE enrichment
            candidates = candidates[: self._retrieval_config.top_k]

        # 4. Context Enrichment - pass source_chunk_map to expanders
        for enricher in self._enrichers:
            if hasattr(enricher, 'expand_with_sources'):
                candidates = enricher.expand_with_sources(candidates, source_chunk_map)
            else:
                candidates = enricher.expand(candidates)

        # 5. Final truncation to retrieval.top_k
        return candidates[: self._retrieval_config.top_k]