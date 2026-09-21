"""Tests for the structure-aware chunker."""

from __future__ import annotations

from rag_sdk.chunking import build_chunker
from rag_sdk.config import StructureAwareChunkerConfig
from rag_sdk.core import Document, DocumentMetadata


def _chunker(**kwargs):
    return build_chunker(StructureAwareChunkerConfig(chunk_size=100, overlap=0, **kwargs))


def test_falls_back_to_recursive_without_headings() -> None:
    document = Document(id="d", text="Hello world. This is a test. " * 10)
    chunks = _chunker().chunk(document)
    assert chunks
    assert all(c.document_id == "d" for c in chunks)


def test_offsets_match_text_and_markdown_prefix_is_consumed() -> None:
    text = "# Intro\n\n   First para here. Second one.\n# Next\nMore text."
    document = Document(
        id="d", text=text, metadata=DocumentMetadata(headings=["Intro", "Next"])
    )
    chunks = _chunker().chunk(document)
    assert [c.text for c in chunks] == ["First para here. Second one.", "More text."]
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
    assert chunks[1].metadata.section_hierarchy == "Intro > Next"


def test_heading_words_in_body_are_not_mistaken_for_headings() -> None:
    text = "Summary of the intro follows.\nIntro\nBody text here."
    document = Document(id="d", text=text, metadata=DocumentMetadata(headings=["Intro"]))
    chunks = _chunker().chunk(document)
    assert [c.text for c in chunks] == ["Summary of the intro follows.", "Body text here."]
