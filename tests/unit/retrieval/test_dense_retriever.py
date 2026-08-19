"""Tests for the dense retriever."""

from __future__ import annotations

from rag_sdk.core import Chunk
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import DenseRetriever

CHUNK_KWARGS = dict(document_id="d0", index=0, start_char=0, end_char=1)


def test_retrieves_most_similar_chunk(hash_embedding) -> None:
    store = FaissVectorStore(dimension=hash_embedding.dimension)
    retriever = DenseRetriever(hash_embedding, store)
    chunks = [
        Chunk(
            id="c0",
            text="cats and kittens and feline whiskers",
            **CHUNK_KWARGS,
        ),
        Chunk(
            id="c1",
            text="banking interest rates and loans",
            **CHUNK_KWARGS,
        ),
        Chunk(
            id="c2",
            text="cute cats sleeping on keyboards",
            **CHUNK_KWARGS,
        ),
    ]
    retriever.add_chunks(chunks)

    results = retriever.search("cat adoption", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id in {"c0", "c2"}
    assert results[0].query == "cat adoption"
    assert results[1].score <= results[0].score


def test_search_returns_chunk_object_and_metadata(hash_embedding) -> None:
    store = FaissVectorStore(dimension=hash_embedding.dimension)
    retriever = DenseRetriever(hash_embedding, store)
    chunk = Chunk(id="only", text="solo text", **CHUNK_KWARGS)
    retriever.add_chunks([chunk])

    (result,) = retriever.search("solo", top_k=1)

    assert result.chunk is chunk
    assert result.chunk.document_id == "d0"