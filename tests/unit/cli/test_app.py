"""Tests for the rag CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from rag_sdk.cli import app

runner = CliRunner()

CONFIG = """
chunking:
  strategy: fixed
  chunk_size: 128
  overlap: 16
"""


def test_init_writes_config(tmp_path) -> None:
    output = tmp_path / "rag.yaml"
    result = runner.invoke(app, ["init", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "chunking" in output.read_text(encoding="utf-8")


def test_init_refuses_overwrite_without_force(tmp_path) -> None:
    output = tmp_path / "rag.yaml"
    output.write_text(CONFIG, encoding="utf-8")
    result = runner.invoke(app, ["init", "--output", str(output)])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == CONFIG


def test_init_overwrites_with_force(tmp_path) -> None:
    output = tmp_path / "rag.yaml"
    output.write_text(CONFIG, encoding="utf-8")
    result = runner.invoke(app, ["init", "--output", str(output), "--force"])
    assert result.exit_code == 0, result.output
    assert "chunking" in output.read_text(encoding="utf-8")


def test_validate_valid_config(tmp_path) -> None:
    config = tmp_path / "rag.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 0, result.output
    assert "Valid configuration" in result.output
    assert "fixed" in result.output


def test_validate_invalid_config(tmp_path) -> None:
    config = tmp_path / "rag.yaml"
    config.write_text("chunking:\n  strategy: bogus\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code != 0
    assert "Invalid configuration" in result.output


def test_evaluate_metrics_from_jsonl(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"retrieved": ["a", "b", "c"], "relevant": ["a"]})
        + "\n"
        + json.dumps({"retrieved": ["x", "y", "z"], "relevant": ["y"]})
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["evaluate", str(results), "--k", "3"])
    assert result.exit_code == 0, result.output
    assert "hit_at_k" in result.output
    assert "mrr" in result.output


def test_evaluate_rejects_invalid_jsonl(tmp_path) -> None:
    results = tmp_path / "bad.jsonl"
    results.write_text("not json\n", encoding="utf-8")
    result = runner.invoke(app, ["evaluate", str(results)])
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output