"""Loading experiment evaluation datasets.

Each JSONL line is ``{"query": ..., "relevant_documents": [doc_id, ...]}``.
Relevance is expressed at document level; the runner expands it to the
chunk level during evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DatasetError(ValueError):
    """Raised when a dataset file cannot be loaded."""


class QuerySample(BaseModel):
    """A single query with its relevant documents."""

    model_config = ConfigDict(frozen=True)

    query: str
    relevant_documents: list[str] = Field(default_factory=list)


def load_queries(path: str | Path) -> list[QuerySample]:
    """Load per-query ground truth from a JSONL file."""
    dataset_path = Path(path)
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Could not read dataset file: {dataset_path}") from exc

    samples: list[QuerySample] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(data, dict):
            raise DatasetError(f"Line {line_number} must be a JSON object")
        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            raise DatasetError(f"Line {line_number}: 'query' must be a non-empty string")
        relevant = data.get("relevant_documents")
        if not isinstance(relevant, list) or not all(
            isinstance(doc_id, str) for doc_id in relevant
        ):
            raise DatasetError(
                f"Line {line_number}: 'relevant_documents' must be a list of strings"
            )
        samples.append(QuerySample(query=query, relevant_documents=relevant))
    if not samples:
        raise DatasetError(f"Dataset file contains no queries: {dataset_path}")
    return samples