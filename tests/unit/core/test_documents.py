"""Tests for core domain types."""

from __future__ import annotations

from rag_sdk.core import Chunk, Document, DocumentMetadata


def test_document_defaults() -> None:
    doc = Document(id="d1", text="hello")
    assert doc.metadata == DocumentMetadata()


def test_document_with_metadata() -> None:
    doc = Document(
        id="d1",
        text="body",
        metadata=DocumentMetadata(source="file.md", title="T", page=3, headings=["A", "B"]),
    )
    assert doc.metadata.page == 3
    assert doc.metadata.headings == ["A", "B"]


def test_chunk_offsets_and_document_id() -> None:
    chunk = Chunk(id="c1", text="abc", document_id="d1", index=0, start_char=0, end_char=3)
    assert chunk.id == "c1"
    assert chunk.document_id == "d1"
    assert chunk.end_char == 3


def test_models_are_serializable() -> None:
    doc = Document(id="d1", text="hello", metadata=DocumentMetadata(source="f.md"))
    data = doc.model_dump()
    restored = Document.model_validate(data)
    assert restored == doc