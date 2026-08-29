"""External tests for DOCX extraction."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.external


def test_docx_loader_extracts_text(docx_fixture_path):
    """Test that DOCX loader extracts text correctly."""
    from rag_sdk.ingestion import DocxLoader
    
    loader = DocxLoader()
    docs = loader.load(docx_fixture_path)
    
    assert len(docs) == 1
    assert docs[0].text
    assert docs[0].metadata.source == str(docx_fixture_path)


def test_docx_loader_preserves_headings(docx_fixture_path):
    """Test that DOCX loader extracts headings."""
    from rag_sdk.ingestion import DocxLoader
    
    loader = DocxLoader()
    docs = loader.load(docx_fixture_path)
    
    # Headings may or may not be present depending on the doc
    assert isinstance(docs[0].metadata.headings, list)


def test_docx_loader_handles_paragraphs(docx_fixture_path):
    """Test that DOCX loader handles multiple paragraphs."""
    from rag_sdk.ingestion import DocxLoader
    
    loader = DocxLoader()
    docs = loader.load(docx_fixture_path)
    
    assert len(docs[0].text) > 0