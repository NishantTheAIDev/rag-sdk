"""Tests for chunking strategies."""

from __future__ import annotations

from rag_sdk.chunking import FixedTokenChunker, RecursiveChunker, build_chunker
from rag_sdk.config import RagConfig
from rag_sdk.core import Document

TEXT = "The quick brown fox. " * 40
DOC = Document(id="doc", text=TEXT)


def _recursive_chunker(**kwargs: object) -> RecursiveChunker:
    cfg = RagConfig.model_validate({"chunking": {"strategy": "recursive", **kwargs}})
    return RecursiveChunker.from_config(cfg.chunking)


def test_recursive_chunks_are_ordered_and_covered() -> None:
    chunker = _recursive_chunker(chunk_size=128, overlap=0)
    chunks = chunker.chunk(DOC)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == TEXT
    for chunk in chunks:
        assert chunk.document_id == "doc"
        assert chunk.text == TEXT[chunk.start_char : chunk.end_char]
        assert chunk.end_char > chunk.start_char
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_recursive_overlap_extends_chunks() -> None:
    chunker = _recursive_chunker(chunk_size=128, overlap=32)
    chunks = chunker.chunk(DOC)
    assert chunks[1].start_char < chunks[0].end_char
    assert chunks[1].start_char == chunks[0].end_char - min(32, len(chunks[0].text))


def test_recursive_single_short_document() -> None:
    doc = Document(id="short", text="tiny")
    chunks = _recursive_chunker(chunk_size=512, overlap=0).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "tiny"


def test_fixed_token_chunker() -> None:
    cfg = RagConfig.model_validate(
        {"chunking": {"strategy": "fixed", "chunk_size": 6, "overlap": 2}}
    )
    chunker = FixedTokenChunker.from_config(cfg.chunking)
    chunks = chunker.chunk(DOC)
    for chunk in chunks:
        assert len(chunk.text.split()) <= 6
    assert chunks[1].start_char < chunks[0].end_char


def test_fixed_token_covers_tail() -> None:
    cfg = RagConfig.model_validate(
        {"chunking": {"strategy": "fixed", "chunk_size": 8, "overlap": 2}}
    )
    chunks = FixedTokenChunker.from_config(cfg.chunking).chunk(DOC)
    last_chunk_text = TEXT[chunks[-1].start_char : chunks[-1].end_char]
    assert last_chunk_text.split()[-1] == "fox."


def test_factory_resolves_strategies() -> None:
    recursive = build_chunker(
        RagConfig.model_validate({"chunking": {"strategy": "recursive"}}).chunking
    )
    fixed = build_chunker(
        RagConfig.model_validate(
            {"chunking": {"strategy": "fixed", "chunk_size": 10, "overlap": 2}}
        ).chunking
    )
    assert isinstance(recursive, RecursiveChunker)
    assert isinstance(fixed, FixedTokenChunker)