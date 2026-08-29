"""Loading evaluation datasets.

Each JSONL line is ``{"query": ..., "relevant_documents": [doc_id, ...]}``.
Relevance is expressed at document level; the runner expands it to the
chunk level during evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_sdk.dataset import EvaluationDataset, QuerySample


class DatasetError(ValueError):
    """Raised when a dataset file cannot be loaded."""


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
        relevant_chunks = data.get("relevant_chunks", [])
        if not isinstance(relevant_chunks, list) or not all(
            isinstance(cid, str) for cid in relevant_chunks
        ):
            raise DatasetError(
                f"Line {line_number}: 'relevant_chunks' must be a list of strings"
            )
        reference_answer = data.get("reference_answer")
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise DatasetError(
                f"Line {line_number}: 'reference_answer' must be a string"
            )
        query_id = data.get("query_id", f"q{line_number}")
        
        # Graded relevance
        relevance_grades = data.get("relevance_grades", {})
        if not isinstance(relevance_grades, dict):
            raise DatasetError(
                f"Line {line_number}: 'relevance_grades' must be an object"
            )
        # Validate grades are integers 0-3
        for chunk_id, grade in relevance_grades.items():
            is_valid = (
                isinstance(chunk_id, str)
                and isinstance(grade, int)
                and 0 <= grade <= 3
            )
            if not is_valid:
                raise DatasetError(
                    f"Line {line_number}: 'relevance_grades' must map chunk IDs to integers 0-3"
                )
        
        samples.append(QuerySample(
            query=query,
            relevant_documents=relevant,
            relevant_chunks=relevant_chunks,
            reference_answer=reference_answer,
            query_id=query_id,
            relevance_grades=relevance_grades,
        ))
    if not samples:
        raise DatasetError(f"Dataset file contains no queries: {dataset_path}")
    return samples


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load a complete evaluation dataset with version and metadata."""
    dataset_path = Path(path)
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Could not read dataset file: {dataset_path}") from exc

    if not lines:
        raise DatasetError(f"Dataset file is empty: {dataset_path}")

    # First line may contain dataset metadata
    first_line = lines[0].strip()
    version = "1.0"
    metadata = {}
    sample_lines = lines

    try:
        first_data = json.loads(first_line)
        if "version" in first_data or "metadata" in first_data:
            version = first_data.get("version", "1.0")
            metadata = first_data.get("metadata", {})
            sample_lines = lines[1:]
    except json.JSONDecodeError:
        pass  # Not a metadata line, treat all lines as samples

    samples = []
    for line_number, line in enumerate(sample_lines, start=1):
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
        relevant_chunks = data.get("relevant_chunks", [])
        if not isinstance(relevant_chunks, list) or not all(
            isinstance(cid, str) for cid in relevant_chunks
        ):
            raise DatasetError(
                f"Line {line_number}: 'relevant_chunks' must be a list of strings"
            )
        reference_answer = data.get("reference_answer")
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise DatasetError(
                f"Line {line_number}: 'reference_answer' must be a string"
            )
        query_id = data.get("query_id", f"q{line_number}")
        
        # Graded relevance
        relevance_grades = data.get("relevance_grades", {})
        if not isinstance(relevance_grades, dict):
            raise DatasetError(
                f"Line {line_number}: 'relevance_grades' must be an object"
            )
        for chunk_id, grade in relevance_grades.items():
            is_valid = (
                isinstance(chunk_id, str)
                and isinstance(grade, int)
                and 0 <= grade <= 3
            )
            if not is_valid:
                raise DatasetError(
                    f"Line {line_number}: 'relevance_grades' must map chunk IDs to integers 0-3"
                )
        
        samples.append(QuerySample(
            query=query,
            relevant_documents=relevant,
            relevant_chunks=relevant_chunks,
            reference_answer=reference_answer,
            query_id=query_id,
            relevance_grades=relevance_grades,
        ))

    if not samples:
        raise DatasetError(f"Dataset file contains no queries: {dataset_path}")

    return EvaluationDataset(
        version=version,
        metadata=metadata,
        samples=samples,
    )