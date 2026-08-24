"""Tests for MMR retriever."""

from __future__ import annotations

from rag_sdk.config import MMRRetrievalConfig
from rag_sdk.core import Chunk
from rag_sdk.embeddings import HashEmbeddingProvider
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import MMRRetriever
from rag_sdk.retrieval.base import RetrievalResult


def _mk_chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        id=f"c{idx}",
        document_id="d1",
        text=text,
        index=idx,
        start_char=0,
        end_char=len(text),
    )


def test_mmr_retriever_basic():
    """Test basic MMR retrieval."""
    config = MMRRetrievalConfig(lambda_param=0.5, top_k=3, candidate_k=10)
    embedding = HashEmbeddingProvider(dimension=32)
    store = FaissVectorStore(dimension=32)

    chunks = [
        _mk_chunk(0, "apple banana cherry"),
        _mk_chunk(1, "apple banana date"),
        _mk_chunk(2, "fig grape honeydew"),
        _mk_chunk(3, "fig grape kiwi"),
        _mk_chunk(4, "lemon mango nectarine"),
    ]

    retriever = MMRRetriever(config, embedding, store)
    retriever.add_chunks(chunks)

    results = retriever.search("fruit", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_mmr_lambda_pure_relevance():
    """Test MMR with lambda=1.0 (pure relevance)."""
    config = MMRRetrievalConfig(lambda_param=1.0, top_k=2, candidate_k=10)
    embedding = HashEmbeddingProvider(dimension=32)
    store = FaissVectorStore(dimension=32)

    chunks = [
        _mk_chunk(0, "machine learning"),
        _mk_chunk(1, "deep learning"),
        _mk_chunk(2, "natural language processing"),
    ]

    retriever = MMRRetriever(config, embedding, store)
    retriever.add_chunks(chunks)

    results = retriever.search("learning", top_k=2)
    assert len(results) == 2
    # Should prioritize relevance over diversity
    assert "learning" in results[0].chunk.text.lower()


def test_mmr_lambda_pure_diversity():
    """Test MMR with lambda=0.0 (pure diversity)."""
    config = MMRRetrievalConfig(lambda_param=0.0, top_k=2, candidate_k=10)
    embedding = HashEmbeddingProvider(dimension=32)
    store = FaissVectorStore(dimension=32)

    chunks = [
        _mk_chunk(0, "apple banana"),
        _mk_chunk(1, "apple cherry"),
        _mk_chunk(2, "fig grape"),
    ]

    retriever = MMRRetriever(config, embedding, store)
    retriever.add_chunks(chunks)

    results = retriever.search("fruit", top_k=2)
    assert len(results) == 2
    # Should prioritize diversity - results should be different
    assert results[0].chunk.id != results[1].chunk.id


def test_mmr_empty_chunks():
    """Test MMR with no chunks."""
    config = MMRRetrievalConfig()
    embedding = HashEmbeddingProvider(dimension=32)
    store = FaissVectorStore(dimension=32)

    retriever = MMRRetriever(config, embedding, store)
    retriever.add_chunks([])

    results = retriever.search("query", top_k=5)
    assert results == []


def test_mmr_top_k_exceeds_chunks():
    """Test MMR when top_k exceeds available chunks."""
    config = MMRRetrievalConfig(top_k=10, candidate_k=10)
    embedding = HashEmbeddingProvider(dimension=32)
    store = FaissVectorStore(dimension=32)

    chunks = [_mk_chunk(i, t) for i, t in enumerate(["first chunk", "second chunk"])]

    retriever = MMRRetriever(config, embedding, store)
    retriever.add_chunks(chunks)

    results = retriever.search("query", top_k=10)
    assert len(results) == 2


def test_mmr_lineage_preserved():
    """Test that lineage fields are preserved in results."""
    config = MMRRetrievalConfig(top_k=3, candidate_k=10)
    embedding = HashEmbeddingProvider(dimension=32)
    store = FaissVectorStore(dimension=32)

    texts = ["test content one", "test content two", "test content three"]
    chunks = [_mk_chunk(i, t) for i, t in enumerate(texts)]

    retriever = MMRRetriever(config, embedding, store)
    retriever.add_chunks(chunks)

    results = retriever.search("test", top_k=3)

    for i, r in enumerate(results):
        assert r.source_chunk_id == r.chunk.id
        assert r.source_chunk_score == r.score
        assert r.source_chunk_rank == i