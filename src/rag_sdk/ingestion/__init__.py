"""Document ingestion."""

from __future__ import annotations

from rag_sdk.ingestion.loader import (
    DocumentLoader,
    DocxLoader,
    HTMLLoader,
    IngestionError,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    build_loader,
    load_documents,
    register_loader,
)
from rag_sdk.ingestion.pipeline import IngestionResult, ingest_documents

__all__ = [
    "IngestionError",
    "IngestionResult",
    "DocumentLoader",
    "TextLoader",
    "JSONLoader",
    "PyPDFLoader",
    "DocxLoader",
    "HTMLLoader",
    "build_loader",
    "register_loader",
    "ingest_documents",
    "load_documents",
]