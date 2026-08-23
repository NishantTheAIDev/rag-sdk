"""Retrieval strategies."""

from __future__ import annotations

from rag_sdk.retrieval.base import RetrievalResult, Retriever
from rag_sdk.retrieval.bm25 import BM25Retriever
from rag_sdk.retrieval.dense import DenseRetriever
from rag_sdk.retrieval.factory import (
    build_retriever,
    register_retriever,
    retriever_registry,
)
from rag_sdk.retrieval.fusion import rrf_fuse, weighted_fuse
from rag_sdk.retrieval.hybrid import HybridRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "Retriever",
    "build_retriever",
    "register_retriever",
    "retriever_registry",
    "rrf_fuse",
    "weighted_fuse",
]