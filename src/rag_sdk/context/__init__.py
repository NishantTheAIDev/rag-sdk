"""Context construction for RAG."""

from __future__ import annotations

from rag_sdk.context.base import Context, ContextBuilder, ContextConfig
from rag_sdk.context.default import DefaultContextBuilder
from rag_sdk.context.factory import build_context_builder, context_builder_registry

__all__ = [
    "ContextBuilder",
    "Context",
    "ContextConfig",
    "DefaultContextBuilder",
    "build_context_builder",
    "context_builder_registry",
    "register_context_builder",
]

register_context_builder = context_builder_registry.decorator