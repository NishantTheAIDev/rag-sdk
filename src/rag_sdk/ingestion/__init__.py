"""Document ingestion."""

from __future__ import annotations

from rag_sdk.ingestion.loader import IngestionError, load_documents
from rag_sdk.ingestion.pipeline import IngestionResult, ingest_documents

__all__ = ["IngestionError", "IngestionResult", "ingest_documents", "load_documents"]