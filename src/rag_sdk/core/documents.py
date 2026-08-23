"""Core domain types shared across the SDK."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document-level metadata preserved through chunking and indexing."""

    source: str | None = None
    title: str | None = None
    page: int | None = None
    headings: list[str] = Field(default_factory=list)
    # Parent-child chunking
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    chunk_type: str | None = None  # "parent" | "child" | None
    # Sentence window expansion
    sentence_boundaries: list[tuple[int, int]] = Field(default_factory=list)


class Document(BaseModel):
    """A source document to be chunked and indexed."""

    id: str
    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class Chunk(BaseModel):
    """A contiguous slice of a document produced by a chunker."""

    id: str
    text: str
    document_id: str
    index: int
    start_char: int
    end_char: int
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)