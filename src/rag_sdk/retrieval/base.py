"""Retrieval interface and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rag_sdk.core import Chunk


class RetrievalResult(BaseModel):
    """A single retrieved chunk for a query."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    query: str
    chunk: Chunk
    score: float

    # Lineage - ALWAYS points to original retrieved chunk
    source_chunk_id: str = ""
    source_chunk_score: float = 0.0
    source_chunk_rank: int = 0

    # Reranker lineage
    rerank_score: float | None = None
    rerank_rank: int | None = None

    # Enrichment tracking
    expansion_type: str = "none"
    parent_id: str | None = None
    child_ids: list[str] = []
    merged_source_ids: list[str] = []


class Retriever(ABC):
    """Retrieves relevant chunks for a query."""

    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Register chunks so the retriever can serve them."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return up to ``top_k`` results ordered by relevance."""