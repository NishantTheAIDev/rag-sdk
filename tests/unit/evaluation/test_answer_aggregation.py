"""Tests for end-to-end evaluation metric aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_sdk.config import RagConfig
from rag_sdk.core import Document
from rag_sdk.evaluation import evaluate_rag

DOCS = [
    Document(id="cats", text="Cats purr when content. Kittens are young cats. " * 3),
    Document(id="astronomy", text="Planets orbit stars. The Milky Way is a galaxy. " * 3),
]

CONFIG = RagConfig.model_validate(
    {
        "chunking": {"strategy": "recursive", "chunk_size": 64, "overlap": 8},
        "retrieval": {"strategy": "bm25", "top_k": 3},
        "generation": {"provider": "mock", "model": "mock"},
        "evaluation": {
            "answer": {"metrics": ["context_precision", "context_recall", "correctness"]}
        },
    }
)


def _dataset(tmp_path: Path, *lines: dict, name: str = "queries.jsonl") -> str:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return str(path)


def test_context_metrics_use_document_labels(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path, {"query": "kittens young cats purr", "relevant_documents": ["cats"]}
    )

    metrics = evaluate_rag(CONFIG, DOCS, dataset)["metrics"]

    assert metrics["context_precision"] > 0.0
    assert metrics["context_recall"] == 1.0


def test_unlabelled_queries_are_left_out_of_the_average(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        {"query": "kittens young cats purr", "relevant_documents": ["cats"]},
        {"query": "planets orbit stars", "relevant_documents": []},
    )
    labelled_only = _dataset(
        tmp_path,
        {"query": "kittens young cats purr", "relevant_documents": ["cats"]},
        name="labelled.jsonl",
    )

    mixed = evaluate_rag(CONFIG, DOCS, dataset)["metrics"]
    single = evaluate_rag(CONFIG, DOCS, labelled_only)["metrics"]

    assert mixed["context_precision"] == pytest.approx(single["context_precision"])
    assert mixed["context_recall"] == pytest.approx(single["context_recall"])


def test_metric_omitted_when_no_query_is_labelled(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, {"query": "planets orbit stars", "relevant_documents": []})

    metrics = evaluate_rag(CONFIG, DOCS, dataset)["metrics"]

    assert "context_precision" not in metrics
    assert "context_recall" not in metrics


def test_correctness_averages_only_queries_with_reference_answers(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        # The mock generator answers "Mock response".
        {"query": "kittens", "relevant_documents": ["cats"], "reference_answer": "Mock response"},
        {"query": "planets", "relevant_documents": ["astronomy"]},
    )

    metrics = evaluate_rag(CONFIG, DOCS, dataset)["metrics"]

    assert metrics["correctness"] == 1.0  # previously 0.5: the unreferenced query counted as 0
