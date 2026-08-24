"""Loading retrieval evaluation results from JSONL files.

Each line is ``{"retrieved": [id, ...], "relevant": [id, ...]}``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rag_sdk.dataset import QuerySample
from rag_sdk.dataset.loader import load_queries


def load_retrieval_results(
    path: str | Path,
) -> list[tuple[Sequence[str], set[str]]]:
    """Load per-query retrieval samples from a JSONL file."""
    results_path = Path(path)
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read results file: {results_path}") from exc

    samples: list[tuple[Sequence[str], set[str]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Line {line_number} must be a JSON object")
        retrieved = data.get("retrieved")
        relevant = data.get("relevant")
        if not isinstance(retrieved, list) or not all(
            isinstance(item, str) for item in retrieved
        ):
            raise ValueError(f"Line {line_number}: 'retrieved' must be a list of strings")
        if not isinstance(relevant, list) or not all(
            isinstance(item, str) for item in relevant
        ):
            raise ValueError(f"Line {line_number}: 'relevant' must be a list of strings")
        samples.append((retrieved, set(relevant)))
    return samples


def load_evaluation_samples(path: str | Path) -> list[QuerySample]:
    """Load evaluation samples (with reference answers) from a JSONL file."""
    return load_queries(path)