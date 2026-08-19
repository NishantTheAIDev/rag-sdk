"""Retrieval strategies."""

from __future__ import annotations

from rag_sdk.retrieval.base import RetrievalResult, Retriever
from rag_sdk.retrieval.dense import DenseRetriever

__all__ = ["DenseRetriever", "RetrievalResult", "Retriever"]