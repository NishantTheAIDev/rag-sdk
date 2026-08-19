"""Chunker factory resolved by configuration."""

from __future__ import annotations

from rag_sdk.chunking.base import Chunker
from rag_sdk.chunking.fixed import FixedTokenChunker
from rag_sdk.chunking.recursive import RecursiveChunker
from rag_sdk.config import ChunkerConfig
from rag_sdk.core import Registry

chunker_registry: Registry[type[Chunker]] = Registry()
chunker_registry.register("recursive", RecursiveChunker)
chunker_registry.register("fixed", FixedTokenChunker)


def build_chunker(config: ChunkerConfig) -> Chunker:
    """Resolve and construct a chunker from its configuration."""
    chunker_type = chunker_registry.get(config.strategy)
    return chunker_type.from_config(config)


register_chunker = chunker_registry.decorator