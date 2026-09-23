"""BM25 (lexical) retrieval backed by ``bm25s``."""

from __future__ import annotations

from collections.abc import Sequence

import bm25s

from rag_sdk.config import BM25Params, BM25RetrievalConfig
from rag_sdk.core import Chunk, MetadataFilters, matches_filters
from rag_sdk.retrieval.base import RetrievalResult, Retriever

_WHITESPACE_PATTERN = r"\S+"


class BM25Retriever(Retriever):
    """Ranks chunks by Okapi BM25 term overlap.

    ``add_chunks`` rebuilds the underlying sparse index, so the corpus must be
    known before ``search`` is called.
    """

    def __init__(
        self,
        config: BM25RetrievalConfig | BM25Params | None = None,
        filters: MetadataFilters | None = None,
    ) -> None:
        params = config or BM25Params()
        # ``filters`` overrides the config's filters (used by hybrid retrieval,
        # whose BM25 side is configured by ``BM25Params`` only).
        if filters is None and isinstance(config, BM25RetrievalConfig):
            filters = config.filters
        self._filters = dict(filters) if filters else {}
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
        # With filters, rank the whole corpus so filtering cannot starve results.
        limit = len(self._chunks) if self._filters else min(top_k, len(self._chunks))
        hits, hit_scores = self._model.retrieve(query_tokens, k=limit, return_as="tuple")

        results: list[RetrievalResult] = []
        for score, chunk_id in zip(hit_scores[0], hits[0], strict=True):
            chunk = self._chunks.get(str(chunk_id))
            if chunk is None:
                continue
            if self._filters and not matches_filters(chunk.metadata, self._filters):
                continue
            results.append(RetrievalResult(query=query, chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def _tokenize(self, texts: Sequence[str]) -> list[list[str]]:
        if self._tokenizer == "whitespace":
            return [
                [token for token in text.split() if token]
                for text in texts
            ]
        stopwords = "english" if self._stopwords else None
        return bm25s.tokenize(list(texts), stopwords=stopwords)