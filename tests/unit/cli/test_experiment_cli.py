"""Tests for the ``rag experiment`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rag_sdk.cli.app import app
from rag_sdk.config import RagConfig
from rag_sdk.config.loader import dump_config, load_config

runner = CliRunner()


def _config(tmp_path: Path, output: Path) -> str:
    config = RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive", "chunk_size": 128, "overlap": 16},
            "retrieval": {"strategy": "dense", "top_k": 5},
            "documents": {"path": str(tmp_path / "docs")},
            "experiments": {
                "dataset": str(tmp_path / "queries.jsonl"),
                "k": 3,
                "output_dir": str(output),
                "parameters": {
                    "retrieval.strategy": ["dense", "hybrid"],
                    "chunking.chunk_size": [128],
                },
            },
        }
    )
    path = tmp_path / "rag.yaml"
    path.write_text(dump_config(config), encoding="utf-8")
    return str(path)


def _corpus(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cats.md").write_text(
        "Cats are small carnivorous mammals. Cats purr when content. "
        "Kittens are young cats. Cat owners value feline companionship. ",
        encoding="utf-8",
    )
    (docs / "astronomy.md").write_text(
        "Astronomy studies celestial objects and the cosmos. Planets orbit stars. "
        "The Milky Way is a barred spiral galaxy. Astronomers use telescopes. ",
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "kittens cats", "relevant_documents": ["cats"]}) + "\n"
        + json.dumps(
            {"query": "planets stars", "relevant_documents": ["astronomy"]}
        ) + "\n",
        encoding="utf-8",
    )


def test_experiment_command_end_to_end(tmp_path: Path) -> None:
    _corpus(tmp_path)
    output = tmp_path / "out"
    config_path = _config(tmp_path, output)

    result = runner.invoke(app, ["experiment", config_path])

    assert result.exit_code == 0, result.output
    assert "Ran 2 configuration(s)" in result.output
    assert "Best by mrr" in result.output
    for name in ("results.csv", "results.json", "leaderboard.csv", "report.html"):
        assert (output / name).exists()


def test_experiment_command_requires_experiments_section(tmp_path: Path) -> None:
    config = RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive"},
            "documents": {"path": str(tmp_path / "docs")},
        }
    )
    path = tmp_path / "rag.yaml"
    path.write_text(dump_config(config), encoding="utf-8")

    result = runner.invoke(app, ["experiment", str(path)])

    assert result.exit_code != 0
    assert "'experiments'" in result.output


def test_experiment_command_invalid_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("chunking: [not, a, mapping]", encoding="utf-8")

    result = runner.invoke(app, ["experiment", str(path)])

    assert result.exit_code != 0

def test_optimize_writes_full_recommended_config(tmp_path: Path) -> None:
    _corpus(tmp_path)
    output = tmp_path / "out"
    config_path = _config(tmp_path, output)
    assert runner.invoke(app, ["experiment", config_path]).exit_code == 0

    result = runner.invoke(app, ["optimize", config_path])

    assert result.exit_code == 0, result.output
    assert "Recommended run: run-" in result.output
    optimized = load_config(output / "optimized-rag.yaml")
    assert optimized.chunking.chunk_size == 128
    assert optimized.retrieval.strategy in {"dense", "hybrid"}
    assert optimized.experiments is None


def test_optimize_without_results_fails(tmp_path: Path) -> None:
    _corpus(tmp_path)
    config_path = _config(tmp_path, tmp_path / "missing")

    result = runner.invoke(app, ["optimize", config_path])

    assert result.exit_code != 0
    assert "rag experiment" in result.output


def test_experiment_command_summarizes_warnings(tmp_path: Path) -> None:
    _corpus(tmp_path)
    config = RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive", "chunk_size": 128, "overlap": 16},
            "retrieval": {"strategy": "dense", "top_k": 2},
            "documents": {"path": str(tmp_path / "docs")},
            "experiments": {
                "dataset": str(tmp_path / "queries.jsonl"),
                "k": 5,
                "output_dir": str(tmp_path / "out"),
            },
        }
    )
    path = tmp_path / "rag.yaml"
    path.write_text(dump_config(config), encoding="utf-8")

    result = runner.invoke(app, ["experiment", str(path)])

    assert result.exit_code == 0, result.output
    assert "Warnings:" in result.output
    assert "retrieval.top_k (2) < experiments.k (5)" in result.output


def _pipeline_config(tmp_path: Path, **extra: object) -> str:
    config = RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive", "chunk_size": 128, "overlap": 16},
            "retrieval": {"strategy": "dense", "top_k": 3},
            "documents": {"path": str(tmp_path / "docs")},
            "generation": {"provider": "mock", "model": "mock"},
            "evaluation": {"answer": {"enabled": True}},
            **extra,
        }
    )
    path = tmp_path / "pipeline.yaml"
    path.write_text(dump_config(config), encoding="utf-8")
    return str(path)


def test_evaluate_pipeline_reports_retrieval_and_answer_metrics(tmp_path: Path) -> None:
    _corpus(tmp_path)
    config_path = _pipeline_config(tmp_path)
    output = tmp_path / "eval"

    result = runner.invoke(
        app,
        ["evaluate-pipeline", config_path, "--dataset", str(tmp_path / "queries.jsonl"),
         "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluated 2 queries over 2 document(s)" in result.output
    assert "Retrieval metrics:" in result.output
    assert "Answer metrics:" in result.output
    assert "retrieval_mean_ms" in result.output
    report = json.loads((output / "evaluation.json").read_text())
    assert report["retrieval_metrics"]["hit_at_k"] == 1.0
    assert report["k"] == 3


def test_evaluate_pipeline_uses_configured_dataset(tmp_path: Path) -> None:
    _corpus(tmp_path)
    config_path = _pipeline_config(
        tmp_path,
        evaluation={"dataset": str(tmp_path / "queries.jsonl"), "k": 2},
    )

    result = runner.invoke(app, ["evaluate-pipeline", config_path])

    assert result.exit_code == 0, result.output
    assert "k=2" in result.output
    assert "Retrieval metrics:" in result.output


def test_evaluate_pipeline_requires_dataset(tmp_path: Path) -> None:
    _corpus(tmp_path)
    config_path = _pipeline_config(tmp_path)

    result = runner.invoke(app, ["evaluate-pipeline", config_path])

    assert result.exit_code != 0
    assert "--dataset" in result.output


def test_evaluate_pipeline_requires_documents(tmp_path: Path) -> None:
    config = RagConfig.model_validate(
        {"chunking": {"strategy": "recursive", "chunk_size": 128, "overlap": 16}}
    )
    path = tmp_path / "rag.yaml"
    path.write_text(dump_config(config), encoding="utf-8")

    result = runner.invoke(
        app, ["evaluate-pipeline", str(path), "--dataset", str(tmp_path / "q.jsonl")]
    )

    assert result.exit_code != 0
    assert "documents.path" in result.output
