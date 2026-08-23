"""Tests for context construction."""

from __future__ import annotations

from rag_sdk.context import ContextConfig, DefaultContextBuilder
from rag_sdk.core import Chunk
from rag_sdk.retrieval.base import RetrievalResult
from rag_sdk.tokenizer.base import WhitespaceTokenizerConfig


def _make_chunk(text: str, chunk_id: str = "c1") -> RetrievalResult:
    chunk = Chunk(
        id=chunk_id,
        document_id="d1",
        text=text,
        index=0,
        start_char=0,
        end_char=len(text),
    )
    return RetrievalResult(query="test", chunk=chunk, score=1.0)


def test_default_context_builder_basic():
    """Test basic context building."""
    config = ContextConfig(max_tokens=200)
    builder = DefaultContextBuilder(config)

    chunks = [
        _make_chunk("First chunk content", "c1"),
        _make_chunk("Second chunk content", "c2"),
    ]

    context = builder.build(chunks, "test query")

    assert context.token_count > 0
    assert "First chunk content" in context.text
    assert "Second chunk content" in context.text
    assert len(context.source_chunks) == 2


def test_context_deduplication():
    """Test that duplicate chunks are removed."""
    config = ContextConfig(max_tokens=1000, deduplicate=True)
    builder = DefaultContextBuilder(config)

    chunk = _make_chunk("Same content", "c1")
    chunks = [chunk, chunk, _make_chunk("Different content", "c2")]

    context = builder.build(chunks, "test query")

    assert len(context.source_chunks) == 2


def test_context_token_budget():
    """Test token budget limits context."""
    config = ContextConfig(max_tokens=20, tokenizer=WhitespaceTokenizerConfig())
    builder = DefaultContextBuilder(config)

    chunks = [
        _make_chunk("This is a very long chunk with many words"),
        _make_chunk("Another long chunk with many words"),
        _make_chunk("Third chunk"),
    ]

    context = builder.build(chunks, "test query")

    assert context.token_count <= 20


def test_context_includes_metadata():
    """Test metadata inclusion."""
    config = ContextConfig(include_metadata=True, tokenizer=WhitespaceTokenizerConfig())
    builder = DefaultContextBuilder(config)

    chunk = _make_chunk("Content", "chunk1")
    context = builder.build([chunk], "test query")

    assert "[d1:chunk1]" in context.text


def test_context_excludes_metadata():
    """Test metadata exclusion."""
    config = ContextConfig(include_metadata=False, tokenizer=WhitespaceTokenizerConfig())
    builder = DefaultContextBuilder(config)

    chunk = _make_chunk("Content", "chunk1")
    context = builder.build([chunk], "test query")

    assert "[d1:chunk1]" not in context.text


def test_context_config_defaults():
    """Test context config defaults."""
    config = ContextConfig()
    assert config.max_tokens == 2048
    assert config.include_metadata is True
    assert config.deduplicate is True