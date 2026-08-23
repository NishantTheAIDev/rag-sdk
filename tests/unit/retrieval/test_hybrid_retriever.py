"""Tests for the hybrid retriever."""

from __future__ import annotations

from rag_sdk.config import HybridRetrievalConfig
from rag_sdk.core import Chunk
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

CHUNK_KWARGS = dict(document_id="d0", index=0, start_char=0, end_char=1)

CORPUS = [
    Chunk(id="c0", text="cats and kittens and feline whiskers", **CHUNK_KWARGS),
    Chunk(id="c1", text="banking interest rates and loans", **CHUNK_KWARGS),
    Chunk(id="c2", text="cute cats sleeping on keyboards", **CHUNK_KWARGS),
]


def _hybrid(hash_embedding, config: HybridRetrievalConfig | None = None) -> HybridRetriever:
    store = FaissVectorStore(dimension=hash_embedding.dimension)
    dense = DenseRetriever(hash_embedding, store)
    lexical = BM25Retriever()
    retriever = HybridRetriever(dense, lexical, config)
    retriever.add_chunks(CORPUS)
    return retriever


def test_rrf_hybrid_retrieves_relevant_chunks(hash_embedding) -> None:
    retriever = _hybrid(hash_embedding)

    results = retriever.search("cats kittens", top_k=3)

    assert len(results) == 3
    assert results[0].chunk.id == "c0"
    assert results[0].score >= results[1].score


def test_weighted_hybrid_retrieves_top_k(hash_embedding) -> None:
    config = HybridRetrievalConfig(
        fusion={"method": "weighted", "dense_weight": 0.4, "bm25_weight": 0.6}
    )
    retriever = _hybrid(hash_embedding, config)

    results = retriever.search("banking loans", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "c1"


def test_top_k_zero_returns_empty(hash_embedding) -> None:
    retriever = _hybrid(hash_embedding)

    assert retriever.search("cats", top_k=0) == []