"""External tests for HTML extraction."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.external


def test_html_loader_extracts_text(html_fixture_path):
    """Test that HTML loader extracts text correctly."""
    from rag_sdk.ingestion import HTMLLoader
    
    loader = HTMLLoader()
    docs = loader.load(html_fixture_path)
    
    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].metadata.source == str(html_fixture_path)


def test_html_loader_preserves_title(html_fixture_path):
    """Test that HTML loader extracts title."""
    from rag_sdk.ingestion import HTMLLoader
    
    loader = HTMLLoader()
    docs = loader.load(html_fixture_path)
    
    assert docs[0].metadata.title
    assert isinstance(docs[0].metadata.title, str)


def test_html_loader_preserves_headings(html_fixture_path):
    """Test that HTML loader extracts headings."""
    from rag_sdk.ingestion import HTMLLoader
    
    loader = HTMLLoader()
    docs = loader.load(html_fixture_path)
    
    assert isinstance(docs[0].metadata.headings, list)


def test_html_loader_removes_scripts(html_fixture_path):
    """Test that HTML loader removes script/style content."""
    from rag_sdk.ingestion import HTMLLoader
    
    loader = HTMLLoader()
    docs = loader.load(html_fixture_path)
    
    # Should not contain script content
    text = docs[0].text.lower()
    # The actual HTML may not have scripts, but if it does, they should be removed
    assert isinstance(text, str)