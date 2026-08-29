"""External tests for PDF extraction."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.external


def test_pdf_loader_extracts_text(pdf_fixture_path):
    """Test that PDF loader extracts text correctly."""
    from rag_sdk.ingestion import PyPDFLoader
    
    loader = PyPDFLoader()
    docs = loader.load(pdf_fixture_path)
    
    assert len(docs) > 0
    assert all(doc.text for doc in docs)
    assert all(doc.metadata.page is not None for doc in docs)
    assert all(doc.metadata.source == str(pdf_fixture_path) for doc in docs)


def test_pdf_loader_preserves_metadata(pdf_fixture_path):
    """Test that PDF loader preserves page metadata."""
    from rag_sdk.ingestion import PyPDFLoader
    
    loader = PyPDFLoader()
    docs = loader.load(pdf_fixture_path)
    
    for doc in docs:
        assert doc.metadata.page is not None
        assert isinstance(doc.metadata.page, int)
        assert doc.metadata.page >= 1


def test_pdf_loader_handles_empty_page(pdf_fixture_path):
    """Test that PDF loader handles pages with no extractable text."""
    from rag_sdk.ingestion import PyPDFLoader
    
    loader = PyPDFLoader()
    docs = loader.load(pdf_fixture_path)
    
    # Should not raise, may skip empty pages
    assert isinstance(docs, list)