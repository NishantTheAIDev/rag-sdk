"""BM25 (lexical) retrieval backed by ``bm25s``."""

from __future__ import annotations

from collections.abc import Sequence

import bm25s

from rag_sdk.config import BM25Params
from rag_sdk.core import Chunk
from rag_sdk.retrieval.base import RetrievalResult, Retriever

_WHITESPACE_PATTERN = r"\S+"


class BM25Retriever(Retriever):
    """Ranks chunks by Okapi BM25 term overlap.

    ``add_chunks`` rebuilds the underlying sparse index, so the corpus must be
    known before ``search`` is called.
    """

    def __init__(self, params: BM25Params | None = None) -> None:
        params = params or BM25Params()
        self._k1 = params.k1
        self._b = params.b
        self._tokenizer = params.tokenizer
        self._stopwords = params.stopwords
        self._chunks: dict[str, Chunk] = {}
        self._model: bm25s.BM25 | None = None

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = {chunk.id: chunk for chunk in chunks}
        if not chunks:
            self._model = None
            return
        corpus = [chunk.text for chunk in chunks]
        tokens = self._tokenize(corpus)
        self._model = bm25s.BM25(k1=self._k1, b=self._b, corpus=list(self._chunks))
        self._model.index(tokens)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._model is None or top_k < 1 or len(self._chunks) == 0:
            return []
        query_tokens = self._tokenize([query])
        hits, hit_scores = self._model.retrieve(
            query_tokens, k=min(top_k, len(self._chunks)), return_as="tuple"
        )
        return [
            RetrievalResult(
                query=query,
                chunk=self._chunks[str(chunk_id)],
                score=float(score),
            )
            for score, chunk_id in zip(hit_scores[0], hits[0], strict=True)
            if chunk_id in self._chunks
        ]

    def _tokenize(self, texts: Sequence[str]) -> list[list[str]]:
        if self._tokenizer == "whitespace":
            return [
                [token for token in text.split() if token]
                for text in texts
            ]
        stopwords = "english" if self._stopwords else None
        return bm25s.tokenize(list(texts), stopwords=stopwords)