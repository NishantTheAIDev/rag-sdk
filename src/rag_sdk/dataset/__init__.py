"""Shared dataset types for evaluation and experiments."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QuerySample(BaseModel):
    """A single query with its relevant documents."""

    model_config = ConfigDict(frozen=True)

    query: str
    relevant_documents: list[str] = Field(default_factory=list)
    relevant_chunks: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    query_id: str | None = None
    # Graded relevance: chunk_id -> relevance grade (0-3)
    relevance_grades: dict[str, int] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    """Complete evaluation dataset with metadata and versioning."""

    version: str = "1.0"
    metadata: dict = Field(default_factory=dict)
    samples: list[QuerySample] = Field(default_factory=list)


__all__ = [
    "QuerySample",
    "EvaluationDataset",
    "DatasetError",
    "load_queries",
    "load_dataset",
]