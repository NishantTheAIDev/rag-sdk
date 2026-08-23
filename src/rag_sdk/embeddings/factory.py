"""Embedding provider factory resolved by configuration."""

from __future__ import annotations

from rag_sdk.config import EmbeddingConfig
from rag_sdk.core import Registry
from rag_sdk.embeddings.base import EmbeddingProvider
from rag_sdk.embeddings.hash import HashEmbeddingProvider

embedding_registry: Registry[type[EmbeddingProvider]] = Registry()
embedding_registry.register("hash", HashEmbeddingProvider)


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Resolve and construct an embedding provider from its configuration."""
    provider_type = embedding_registry.get(config.provider)
    kwargs: dict[str, object] = {}
    if config.dimension is not None:
        kwargs["dimension"] = config.dimension
    return provider_type(**kwargs)


register_embedding = embedding_registry.decorator