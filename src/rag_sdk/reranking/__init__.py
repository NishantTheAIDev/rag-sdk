"""Reranking providers."""

from __future__ import annotations

from rag_sdk.reranking.base import RerankerProvider
from rag_sdk.reranking.cohere import CohereReranker
from rag_sdk.reranking.cross_encoder import CrossEncoderReranker
from rag_sdk.reranking.factory import build_reranker, register_reranker, reranker_registry
from rag_sdk.reranking.noop import NoOpReranker

__all__ = [
    "CrossEncoderReranker",
    "CohereReranker",
    "NoOpReranker",
    "RerankerProvider",
    "build_reranker",
    "register_reranker",
    "reranker_registry",
]