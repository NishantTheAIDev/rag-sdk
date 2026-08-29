"""BM25 (lexical) retrieval backed by ``bm25s``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import bm25s

from rag_sdk.config import BM25Params, BM25RetrievalConfig
from rag_sdk.core import Chunk
from rag_sdk.retrieval.base import RetrievalResult, Retriever

if TYPE_CHECKING:
    pass

_WHITESPACE_PATTERN = r"\S+"


class BM25Retriever(Retriever):
    """Ranks chunks by Okapi BM25 term overlap.

    ``add_chunks`` rebuilds the underlying sparse index, so the corpus must be
    known before ``search`` is called.
    """

    def __init__(
        self,
        config: BM25RetrievalConfig | BM25Params | None = None,
    ) -> None:
        if isinstance(config, BM25RetrievalConfig):
            self._config = config
            # BM25RetrievalConfig inherits from BM25Params, so params are directly in config
            params = BM25Params(
                k1=config.k1,
                b=config.b,
                tokenizer=config.tokenizer,
                stopwords=config.stopwords,
            )
        elif isinstance(config, BM25Params):
            self._config = None
            params = config
        else:
            self._config = None
            params = BM25Params()
        
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
        # Search more candidates to account for filtering
        candidate_k = self._config.candidate_k if self._config else top_k
        hits, hit_scores = self._model.retrieve(
            query_tokens, k=min(candidate_k, len(self._chunks)), return_as="tuple"
        )
        
        results = []
        for score, chunk_id in zip(hit_scores[0], hits[0], strict=True):
            if chunk_id not in self._chunks:
                continue
            chunk = self._chunks[str(chunk_id)]
            
            # Apply metadata filters
            if self._config and self._config.filters and not self._matches_filters(
                chunk.metadata, self._config.filters
            ):
                continue
            
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break
        
        return results

    def _matches_filters(
        self,
        metadata,
        filters: dict[str, str | int | float | bool | list[str] | list[int]],
    ) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if not hasattr(metadata, key):
                return False
            meta_value = getattr(metadata, key)
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            elif meta_value != value:
                return False
        return True

    def _tokenize(self, texts: Sequence[str]) -> list[list[str]]:
        if self._tokenizer == "whitespace":
            return [
                [token for token in text.split() if token]
                for text in texts
            ]
        stopwords = "english" if self._stopwords else None
        return bm25s.tokenize(list(texts), stopwords=stopwords)