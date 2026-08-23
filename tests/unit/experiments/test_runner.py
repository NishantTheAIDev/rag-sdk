"""Tests for the experiment runner."""

from __future__ import annotations

import pytest

from rag_sdk.config import RagConfig
from rag_sdk.core import Document
from rag_sdk.experiments import ExperimentRunner, run_experiment

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
    Document(
        id="finance",
        text="Finance covers banking, interest rates, and investments. Stocks "
        "rise and fall. Portfolios balance risk and return. Savers earn compound interest. ",
    ),
]

BASE = RagConfig.model_validate(
    {
        "chunking": {"strategy": "recursive", "chunk_size": 128, "overlap": 16},
        "retrieval": {"strategy": "dense", "top_k": 5},
    }
)


def _write_dataset(tmp_path, lines: list[str]) -> str:
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _base_with_experiments(experiments: dict) -> RagConfig:
    data = BASE.model_dump(mode="json")
    data["experiments"] = experiments
    return RagConfig.model_validate(data)


def test_run_produces_records(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            '{"query": "kittens are young cats", "relevant_documents": ["cats"]}',
            '{"query": "planets orbit stars", "relevant_documents": ["astronomy"]}',
        ],
    )
    base = _base_with_experiments(
        {"dataset": dataset, "k": 3, "output_dir": str(tmp_path / "out")}
    )

    result = run_experiment(DOCS, base)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.total_chunks > 0
    assert record.metrics["hit_at_k"] == 1.0
    assert record.metrics["mrr"] == 1.0
    assert record.latency_ms.mean_ms >= 0
    assert record.latency_ms.median_ms >= 0
    assert record.dataset_hash
    assert record.embedding.provider == "hash"
    assert record.timestamp is not None


def test_run_sweeps_parameters(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 2,
            "parameters": {
                "retrieval.strategy": ["dense", "bm25", "hybrid"],
                "chunking.chunk_size": [64, 128],
            },
        }
    )

    result = run_experiment(DOCS, base)

    assert len(result.records) == 6
    strategies = {record.config["retrieval"]["strategy"] for record in result.records}
    assert strategies == {"dense", "bm25", "hybrid"}
    chunk_sizes = {record.config["chunking"]["chunk_size"] for record in result.records}
    assert chunk_sizes == {64, 128}


def test_leaderboard_sorted_by_primary_metric(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            '{"query": "kittens cats", "relevant_documents": ["cats"]}',
            '{"query": "stars planets", "relevant_documents": ["astronomy"]}',
        ],
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 3,
            "primary_metric": "recall_at_k",
            "parameters": {
                "retrieval.strategy": ["dense", "hybrid"],
                "chunking.chunk_size": [128, 256],
            },
        }
    )

    result = run_experiment(DOCS, base)
    leaderboard = result.leaderboard()

    values = [
        record.metrics[result.primary_metric] for record in leaderboard
    ]
    assert values == sorted(values, reverse=True)


def test_runner_records_skipped_parameters(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 2,
            "parameters": {
                "retrieval.strategy": ["dense", "hybrid"],
                "retrieval.fusion.method": ["weighted"],
            },
        }
    )

    with pytest.warns():
        result = run_experiment(DOCS, base)

    records = {record.config["retrieval"]["strategy"]: record for record in result.records}
    assert records["dense"].skipped_parameters == ["retrieval.fusion.method"]
    assert records["hybrid"].skipped_parameters == []
    assert records["hybrid"].config["retrieval"]["fusion"]["method"] == "weighted"


def test_missing_dataset_raises(tmp_path) -> None:
    base = _base_with_experiments({"dataset": str(tmp_path / "missing.jsonl")})
    with pytest.raises(FileNotFoundError):
        run_experiment(DOCS, base)


def test_no_experiments_section_raises() -> None:
    with pytest.raises(ValueError):
        run_experiment(DOCS, BASE)


def test_runner_rejects_empty_documents(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ['{"query": "q", "relevant_documents": []}'])
    base = _base_with_experiments({"dataset": dataset})
    with pytest.raises(ValueError):
        ExperimentRunner([], base, base.experiments).run()