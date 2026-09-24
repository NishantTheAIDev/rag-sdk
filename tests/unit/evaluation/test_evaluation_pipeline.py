"""Tests for end-to-end evaluation and shared relevance judging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_sdk.config import RagConfig
from rag_sdk.core import Chunk, Document
from rag_sdk.dataset import QuerySample
from rag_sdk.evaluation import evaluate_rag, write_evaluation_report
from rag_sdk.evaluation.relevance import judge_retrieval
from rag_sdk.retrieval import RetrievalResult

DOCS = [
    Document(
        id="cats",
        text="Cats are small carnivorous mammals. Cats purr when content. "
        "Kittens are young cats. Cat owners value feline companionship. ",
    ),
    Document(
        id="astronomy",
        text="Astronomy studies celestial objects and the cosmos. Planets orbit "
        "stars. The Milky Way is a barred spiral galaxy. Astronomers use telescopes. ",
    ),
]


def _config(**extra: object) -> RagConfig:
    return RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive", "chunk_size": 64, "overlap": 8},
            "retrieval": {"strategy": "bm25", "top_k": 3},
            "generation": {"provider": "mock", "model": "mock"},
            "evaluation": {"answer": {"enabled": True}},
            **extra,
        }
    )


def _dataset(tmp_path: Path) -> str:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "kittens are young cats",
                "relevant_documents": ["cats"],
                "reference_answer": "Mock response",
            }
        )
        + "\n"
        + json.dumps({"query": "planets orbit stars", "relevant_documents": ["astronomy"]})
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _result(chunk_id: str, document_id: str) -> RetrievalResult:
    chunk = Chunk(
        id=chunk_id, document_id=document_id, text="t", index=0, start_char=0, end_char=1
    )
    return RetrievalResult(chunk=chunk, score=1.0, query="q")


def test_judge_document_level_collapses_chunks() -> None:
    sample = QuerySample(query="q", relevant_documents=["a"])
    retrieved = [_result("a:0", "a"), _result("a:1", "a"), _result("b:0", "b")]

    ranked, relevant = judge_retrieval(retrieved, sample, {}, "document")

    assert ranked == ["a", "b"]
    assert relevant == {"a"}


def test_judge_chunk_level_prefers_grades_then_chunks_then_documents() -> None:
    retrieved = [_result("a:0", "a")]
    by_document = {"a": ["a:0", "a:1"]}

    graded = QuerySample(
        query="q", relevant_documents=["a"], relevant_chunks=["a:1"], relevance_grades={"a:0": 3}
    )
    labelled = QuerySample(query="q", relevant_documents=["a"], relevant_chunks=["a:1"])
    unlabelled = QuerySample(query="q", relevant_documents=["a"])

    assert judge_retrieval(retrieved, graded, by_document, "chunk")[1] == {"a:0": 3, "a:1": 1}
    assert judge_retrieval(retrieved, labelled, by_document, "chunk")[1] == {"a:1"}
    assert judge_retrieval(retrieved, unlabelled, by_document, "chunk")[1] == {"a:0", "a:1"}


def test_evaluate_rag_reports_retrieval_answer_and_run_metadata(tmp_path: Path) -> None:
    result = evaluate_rag(_config(), DOCS, _dataset(tmp_path))

    assert result["k"] == 3
    assert result["relevance_level"] == "document"
    assert result["retrieval_metrics"]["hit_at_k"] == 1.0
    assert result["retrieval_metrics"]["mrr"] == 1.0
    assert "correctness" in result["answer_metrics"]
    assert result["metrics"] == result["answer_metrics"]  # backwards compatible
    assert result["latency_ms"]["retrieval_mean_ms"] >= 0
    assert result["latency_ms"]["total_mean_ms"] >= result["latency_ms"]["retrieval_mean_ms"]
    assert result["dataset"]["queries"] == 2
    assert len(result["dataset"]["hash"]) == 64
    assert result["models"]["generator"] == "mock:mock"
    assert result["models"]["embedding"] == "hash:default"
    assert result["timestamp"]


def test_evaluate_rag_k_and_level_from_config(tmp_path: Path) -> None:
    config = _config(
        evaluation={"k": 1, "relevance_level": "chunk", "answer": {"enabled": True}}
    )

    result = evaluate_rag(config, DOCS, _dataset(tmp_path))

    assert result["k"] == 1
    assert result["relevance_level"] == "chunk"
    # Documents span several chunks; one retrieved chunk cannot recall them all.
    assert result["retrieval_metrics"]["recall_at_k"] < 1.0

    overridden = evaluate_rag(config, DOCS, _dataset(tmp_path), k=2, relevance_level="document")
    assert overridden["k"] == 2
    assert overridden["retrieval_metrics"]["recall_at_k"] == 1.0


def test_write_evaluation_report_is_reproducible_record(tmp_path: Path) -> None:
    result = evaluate_rag(_config(), DOCS, _dataset(tmp_path))

    path = write_evaluation_report(result, tmp_path / "out" / "evaluation.json")
    report = json.loads(path.read_text())

    for key in ("config", "dataset", "retrieval_metrics", "answer_metrics",
                "latency_ms", "timestamp", "models"):
        assert key in report
    RagConfig.model_validate(report["config"])
    first, second = report["queries"]
    assert first["answer"] == "Mock response"
    assert first["retrieved"][0]["document_id"] == "cats"
    assert first["answer_scores"]["correctness"] == 1.0
    # The second query has no reference answer: unmeasured, not scored 0.
    assert second["answer_scores"]["correctness"] is None
    assert report["answer_metrics"]["correctness"] == 1.0


def test_evaluate_rag_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate_rag(_config(), DOCS, str(tmp_path / "missing.jsonl"))
