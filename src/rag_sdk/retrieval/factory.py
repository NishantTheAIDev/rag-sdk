"""Retriever factory resolved by configuration."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.config import (
    BM25RetrievalConfig,
    DenseRetrievalConfig,
    HybridRetrievalConfig,
    MMRRetrievalConfig,
    RagConfig,
    RetrievalConfig,
)
from rag_sdk.core import Chunk, Registry
from rag_sdk.embeddings import EmbeddingProvider
from rag_sdk.generation import build_generator
from rag_sdk.indexing import ChunkStore, DocumentStore, VectorStore
from rag_sdk.reranking import build_reranker
from rag_sdk.retrieval.base import Retriever
from rag_sdk.retrieval.bm25 import BM25Retriever
from rag_sdk.retrieval.dense import DenseRetriever
from rag_sdk.retrieval.hybrid import HybridRetriever
from rag_sdk.retrieval.mmr import MMRRetriever
from rag_sdk.retrieval.multi_query import MultiQueryRetriever
from rag_sdk.retrieval.pipeline import RetrievalPipeline
from rag_sdk.retrieval.query_rewriting import QueryRewriterFactory

retriever_registry: Registry[type[Retriever]] = Registry()
retriever_registry.register("dense", DenseRetriever)
retriever_registry.register("bm25", BM25Retriever)
retriever_registry.register("hybrid", HybridRetriever)
retriever_registry.register("mmr", MMRRetriever)


def build_retriever(
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider,
    store: VectorStore,
) -> Retriever:
    """Construct a base retriever from its configuration.

    The result still requires ``add_chunks`` before ``search``.
    """
    match config:
        case DenseRetrievalConfig():
            return DenseRetriever(embedding_provider, store, config)
        case BM25RetrievalConfig():
            return BM25Retriever(config)
        case HybridRetrievalConfig():
            dense = DenseRetriever(embedding_provider, store, config)
            lexical = BM25Retriever(config.bm25)
            return HybridRetriever(dense, lexical, config)
        case MMRRetrievalConfig():
            return MMRRetriever(config, embedding_provider, store)
        case _:  # pragma: no cover - discriminated union guarantees exhaustion
            raise ValueError(f"Unsupported retrieval strategy: {config.strategy!r}")


def _maybe_wrap_multi_query(
    retriever: Retriever,
    config: RetrievalConfig,
    generation_config,
) -> Retriever:
    """Wrap retriever with MultiQueryRetriever if enabled."""
    if config.multi_query.enabled:
        generator = None
        if config.multi_query.query_generator == "llm" and generation_config:
            generator = build_generator(generation_config)
        return MultiQueryRetriever(retriever, config.multi_query, generator)
    return retriever


def _build_query_rewriter(config: RetrievalConfig, generation_config):
    """Build query rewriter if enabled."""
    if config.query_rewriter.enabled:
        generator = None
        if config.query_rewriter.strategy in ("llm", "hyde") and generation_config:
            generator = build_generator(generation_config)
        return QueryRewriterFactory.create(config.query_rewriter, generator)
    return None


class QueryRewritingRetriever(Retriever):
    """Wrapper that applies query rewriting before retrieval."""

    def __init__(self, base_retriever: Retriever, rewriter):
        self._base_retriever = base_retriever
        self._rewriter = rewriter

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._base_retriever.add_chunks(chunks)

    def search(self, query: str, top_k: int) -> list:
        rewritten_query = self._rewriter.rewrite(query)
        return self._base_retriever.search(rewritten_query, top_k)


def build_retrieval_pipeline(
    config: RagConfig,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunk_store: ChunkStore | None,
    document_store: DocumentStore | None,
    chunks: Sequence[Chunk] | None = None,
) -> RetrievalPipeline:
    """Build the full retrieval pipeline with reranking and enrichment."""
    # Build base retriever
    retriever = build_retriever(config.retrieval, embedding_provider, vector_store)

    # Wrap with query rewriter if enabled
    rewriter = _build_query_rewriter(config.retrieval, config.generation)
    if rewriter:
        retriever = QueryRewritingRetriever(retriever, rewriter)

    # Wrap with multi-query if enabled
    retriever = _maybe_wrap_multi_query(retriever, config.retrieval, config.generation)

    # Add chunks to retriever if provided
    if chunks:
        retriever.add_chunks(chunks)

    # Build reranker
    reranker = build_reranker(config.reranker) if config.reranker else None

    # Build pipeline
    pipeline = RetrievalPipeline(
        retriever=retriever,
        vector_store=vector_store,
        chunk_store=chunk_store,
        document_store=document_store,
        reranker=reranker,
        retrieval_config=config.retrieval,
        reranker_config=config.reranker,
    )
    return pipeline


register_retriever = retriever_registry.decorator