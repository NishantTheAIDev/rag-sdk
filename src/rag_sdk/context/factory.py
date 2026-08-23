"""Context builder factory."""

from __future__ import annotations

from rag_sdk.context import ContextBuilder, ContextConfig
from rag_sdk.context.default import DefaultContextBuilder
from rag_sdk.core import Registry

context_builder_registry: Registry[type[ContextBuilder]] = Registry()
context_builder_registry.register("default", DefaultContextBuilder)


def build_context_builder(config: ContextConfig) -> ContextBuilder:
    """Construct a context builder from its configuration."""
    return DefaultContextBuilder(config)


def register_context_builder(name: str, builder_class: type[ContextBuilder]) -> None:
    """Register a custom context builder."""
    context_builder_registry.register(name, builder_class)