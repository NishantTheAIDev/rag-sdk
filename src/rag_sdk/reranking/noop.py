"""No-op reranker for baseline comparison."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.reranking.base import RerankerProvider
from rag_sdk.retrieval.base import RetrievalResult


class NoOpReranker(RerankerProvider):
    """Reranker that returns candidates unchanged (baseline)."""

    def __init__(self) -> None:
        self._model_name = "none"

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        # Return top_k unchanged
        return list(candidates[:top_k])

    @property
    def model_name(self) -> str:
        return self._model_name