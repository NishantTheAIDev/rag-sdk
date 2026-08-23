"""Retriever factory resolved by configuration."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.config import (
    BM25RetrievalConfig,
    DenseRetrievalConfig,
    HybridRetrievalConfig,
    RagConfig,
    RetrievalConfig,
)
from rag_sdk.core import Chunk, Registry
from rag_sdk.embeddings import EmbeddingProvider
from rag_sdk.indexing import ChunkStore, DocumentStore, VectorStore
from rag_sdk.reranking import build_reranker
from rag_sdk.retrieval.base import Retriever
from rag_sdk.retrieval.bm25 import BM25Retriever
from rag_sdk.retrieval.dense import DenseRetriever
from rag_sdk.retrieval.hybrid import HybridRetriever
from rag_sdk.retrieval.pipeline import RetrievalPipeline

retriever_registry: Registry[type[Retriever]] = Registry()
retriever_registry.register("dense", DenseRetriever)
retriever_registry.register("bm25", BM25Retriever)
retriever_registry.register("hybrid", HybridRetriever)


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
            return DenseRetriever(embedding_provider, store)
        case BM25RetrievalConfig():
            return BM25Retriever(config)
        case HybridRetrievalConfig():
            dense = DenseRetriever(embedding_provider, store)
            lexical = BM25Retriever(config.bm25)
            return HybridRetriever(dense, lexical, config)
        case _:  # pragma: no cover - discriminated union guarantees exhaustion
            raise ValueError(f"Unsupported retrieval strategy: {config.strategy!r}")


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