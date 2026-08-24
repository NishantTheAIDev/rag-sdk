"""Loading experiment evaluation datasets.

Each JSONL line is ``{"query": ..., "relevant_documents": [doc_id, ...]}``.
Relevance is expressed at document level; the runner expands it to the
chunk level during evaluation.
"""

from __future__ import annotations

from rag_sdk.dataset import QuerySample
from rag_sdk.dataset.loader import DatasetError, load_queries

# Re-export for backwards compatibility
__all__ = ["DatasetError", "QuerySample", "load_queries"]