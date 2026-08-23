"""Tests for the BM25 retriever."""

from __future__ import annotations

import pytest

from rag_sdk.config import BM25Params
from rag_sdk.core import Chunk
from rag_sdk.retrieval import BM25Retriever

CHUNK_KWARGS = dict(document_id="d0", index=0, start_char=0, end_char=1)

CORPUS = [
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


def test_retrieves_lexically_relevant_chunks() -> None:
    retriever = BM25Retriever()
    retriever.add_chunks(CORPUS)

    results = retriever.search("cats kittens", top_k=2)

    assert {result.chunk.id for result in results} == {"c0", "c2"}
    assert results[0].chunk.id == "c0"
    assert results[0].score >= 0


def test_top_k_is_respected() -> None:
    retriever = BM25Retriever()
    retriever.add_chunks(CORPUS)

    results = retriever.search("cats kittens", top_k=1)

    assert len(results) == 1


def test_search_before_add_chunks_returns_empty() -> None:
    retriever = BM25Retriever()

    assert retriever.search("cats", top_k=5) == []


def test_empty_corpus_returns_empty() -> None:
    retriever = BM25Retriever()
    retriever.add_chunks([])

    assert retriever.search("cats", top_k=5) == []


def test_whitespace_tokenizer_ignores_stopwords_off() -> None:
    params = BM25Params(tokenizer="whitespace", stopwords=False)
    retriever = BM25Retriever(params)
    retriever.add_chunks(CORPUS)

    results = retriever.search("cats kittens", top_k=3)

    assert results[0].chunk.id == "c0"


def test_bm25_params_validated() -> None:
    with pytest.raises(ValueError):
        BM25Params(k1=0)
    with pytest.raises(ValueError):
        BM25Params(b=1.5)