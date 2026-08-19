"""Retrieval interface and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from rag_sdk.core import Chunk


class RetrievalResult(BaseModel):
    """A single retrieved chunk for a query."""

    model_config = ConfigDict(frozen=True)

    query: str
    chunk: Chunk
    score: float


class Retriever(ABC):
    """Retrieves relevant chunks for a query."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` results ordered by relevance."""