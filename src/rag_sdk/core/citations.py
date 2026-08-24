"""Citation support for RAG answers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """Citation linking answer text to source chunks."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    source_uri: str | None = None
    text_span: tuple[int, int] | None = None
    score: float = 0.0


class CitedAnswer(BaseModel):
    """Answer with inline citations."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)