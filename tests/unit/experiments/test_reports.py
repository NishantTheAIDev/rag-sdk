"""Tests for experiment report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_sdk.config import RagConfig
from rag_sdk.experiments import (
    ExperimentResult,
    write_csv,
    write_html,
    write_json,
    write_leaderboard,
    write_reports,
)
from rag_sdk.experiments.records import (
    EmbeddingInfo,
    ExperimentRecord,
    LatencyStats,
)


def _record(run_id: str, mrr: float, recall: float) -> ExperimentRecord:
    config = RagConfig.model_validate(
        {
            "chunking": {"strategy": "recursive", "chunk_size": 128},
            "retrieval": {"strategy": "dense", "top_k": 5},
        }
    )
    return ExperimentRecord(
        run_id=run_id,
        config=config.model_dump(mode="json"),
        dataset_path="queries.jsonl",
        dataset_hash="abc123",
        timestamp="2026-01-01T00:00:00+00:00",
        embedding=EmbeddingInfo(provider="hash", model=None, dimension=128),
        latency_ms=LatencyStats(mean_ms=1.0, median_ms=0.9),
        metrics={"hit_at_k": 1.0, "precision_at_k": 0.5, "recall_at_k": recall,
                 "mrr": mrr, "ndcg_at_k": 0.8, "map": 0.7},
        total_chunks=10,
        skipped_parameters=["retrieval.fusion.method"] if run_id == "run-0" else [],
    )


def _result() -> ExperimentResult:
    return ExperimentResult(
        dataset_path="queries.jsonl",
        dataset_hash="abc123",
        primary_metric="mrr",
        k=3,
        records=[
            _record("run-0", mrr=0.5, recall=0.4),
            _record("run-1", mrr=0.9, recall=0.8),
        ],
    )


def test_write_csv(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    write_csv(_result(), path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-0"
    assert rows[0]["chunking"] == "recursive"
    assert rows[0]["retrieval"] == "dense"
    assert rows[0]["skipped_parameters"] == "retrieval.fusion.method"
    assert rows[1]["skipped_parameters"] == ""


def test_write_json(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    write_json(_result(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["primary_metric"] == "mrr"
    assert len(payload["records"]) == 2
    assert payload["recommended"]["run_id"] == "run-1"
    assert payload["leaderboard"][0]["run_id"] == "run-1"
    assert "config_yaml" in payload["recommended"]
    record_by_id = {record["run_id"]: record for record in payload["records"]}
    assert record_by_id["run-0"]["skipped_parameters"] == ["retrieval.fusion.method"]
    assert record_by_id["run-1"]["skipped_parameters"] == []


def test_write_leaderboard(tmp_path: Path) -> None:
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(_result(), path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["rank"] == "1"
    assert rows[0]["run_id"] == "run-1"
    assert rows[1]["run_id"] == "run-0"


def test_write_html(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    write_html(_result(), path)
    html = path.read_text(encoding="utf-8")
    assert "RAG SDK Experiment Report" in html
    assert "__DATA__" not in html
    assert "Recommended configuration" in html
    assert "run-1" in html


def test_write_reports_creates_all_files(tmp_path: Path) -> None:
    paths = write_reports(_result(), tmp_path / "out")
    assert set(paths) == {"csv", "json", "leaderboard", "html"}
    for path in paths.values():
        assert path.exists()