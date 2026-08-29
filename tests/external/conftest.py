"""External test configuration and fixtures.

These tests require external files (PDF, DOCX, HTML) that are downloaded
separately and not committed to the repository.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def pdf_fixture_path() -> Path:
    """Path to sample PDF fixture."""
    return FIXTURES_DIR / "sample.pdf"


@pytest.fixture(scope="session")
def docx_fixture_path() -> Path:
    """Path to sample DOCX fixture."""
    return FIXTURES_DIR / "sample.docx"


@pytest.fixture(scope="session")
def html_fixture_path() -> Path:
    """Path to sample HTML fixture."""
    return FIXTURES_DIR / "sample.html"


@pytest.fixture(scope="session")
def long_pdf_fixture_path() -> Path:
    """Path to long PDF fixture for chunking tests."""
    return FIXTURES_DIR / "long_document.pdf"