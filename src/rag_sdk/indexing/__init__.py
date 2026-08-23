"""Vector stores and indexing."""

from __future__ import annotations

from rag_sdk.indexing.base import VectorStore
from rag_sdk.indexing.faiss_store import FaissVectorStore
from rag_sdk.indexing.sqlite_stores import (
    ChunkStore,
    DocumentStore,
    InMemoryChunkStore,
    InMemoryDocumentStore,
    SqliteChunkStore,
    SqliteDocumentStore,
)
from rag_sdk.indexing.stores import ChunkStore as ChunkStoreBase
from rag_sdk.indexing.stores import DocumentStore as DocumentStoreBase

__all__ = [
    "ChunkStore",
    "ChunkStoreBase",
    "DocumentStore",
    "DocumentStoreBase",
    "FaissVectorStore",
    "InMemoryChunkStore",
    "InMemoryDocumentStore",
    "SqliteChunkStore",
    "SqliteDocumentStore",
    "VectorStore",
]