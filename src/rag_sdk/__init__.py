"""RAG SDK.

Configuration-driven RAG experimentation, evaluation, and deployment.
"""

from __future__ import annotations

from rag_sdk.core import Chunk, Document, DocumentMetadata, Registry

__version__ = "0.1.0"

__all__ = ["Chunk", "Document", "DocumentMetadata", "Registry", "__version__"]