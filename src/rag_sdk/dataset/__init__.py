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