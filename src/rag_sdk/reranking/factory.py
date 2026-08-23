"""Reranker factory resolved by configuration."""

from __future__ import annotations

from rag_sdk.config import (
    NoRerankerConfig,
    RerankerConfig,
)
from rag_sdk.core import Registry
from rag_sdk.reranking.base import RerankerProvider
from rag_sdk.reranking.cohere import CohereReranker
from rag_sdk.reranking.cross_encoder import CrossEncoderReranker
from rag_sdk.reranking.noop import NoOpReranker

reranker_registry: Registry[type[RerankerProvider]] = Registry()
reranker_registry.register("cross_encoder", CrossEncoderReranker)
reranker_registry.register("cohere", CohereReranker)
reranker_registry.register("none", NoOpReranker)


def build_reranker(config: RerankerConfig | None) -> RerankerProvider | None:
    """Construct a reranker from its configuration."""
    if config is None:
        return None
    if isinstance(config, NoRerankerConfig):
        return NoOpReranker()
    reranker_type = reranker_registry.get(config.strategy)
    return reranker_type.from_config(config)


def register_reranker(name: str, cls: type[RerankerProvider]) -> None:
    """Register a custom reranker."""
    reranker_registry.register(name, cls)