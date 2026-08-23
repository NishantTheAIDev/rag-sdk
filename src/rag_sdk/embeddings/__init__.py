"""Embedding providers."""

from __future__ import annotations

from rag_sdk.embeddings.base import EmbeddingProvider
from rag_sdk.embeddings.factory import (
    build_embedding_provider,
    embedding_registry,
    register_embedding,
)
from rag_sdk.embeddings.hash import DEFAULT_DIMENSION, HashEmbeddingProvider

__all__ = [
    "DEFAULT_DIMENSION",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "build_embedding_provider",
    "embedding_registry",
    "register_embedding",
]