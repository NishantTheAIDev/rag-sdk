"""Retrieval strategies."""

from __future__ import annotations

from rag_sdk.retrieval.auto_merging import AutoMerger
from rag_sdk.retrieval.base import RetrievalResult, Retriever
from rag_sdk.retrieval.bm25 import BM25Retriever
from rag_sdk.retrieval.dense import DenseRetriever
from rag_sdk.retrieval.factory import (
    build_retrieval_pipeline,
    build_retriever,
    register_retriever,
    retriever_registry,
)
from rag_sdk.retrieval.fusion import rrf_fuse, weighted_fuse
from rag_sdk.retrieval.hybrid import HybridRetriever
from rag_sdk.retrieval.parent_child import ParentChildExpander
from rag_sdk.retrieval.pipeline import RetrievalPipeline
from rag_sdk.retrieval.sentence_window import SentenceWindowExpander

__all__ = [
    "AutoMerger",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "ParentChildExpander",
    "RetrievalPipeline",
    "RetrievalResult",
    "Retriever",
    "SentenceWindowExpander",
    "build_retrieval_pipeline",
    "build_retriever",
    "register_retriever",
    "retriever_registry",
    "rrf_fuse",
    "weighted_fuse",
]