"""Reranking provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_sdk.retrieval.base import RetrievalResult


class RerankerProvider(ABC):
    """Reranks retrieval results for a query."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank candidates and return top_k results."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...