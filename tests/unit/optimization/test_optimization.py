"""Tests for loading experiment results and Pareto optimization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_sdk.config import RagConfig
from rag_sdk.optimization import ParetoOptimizer, load_experiment_records

CONFIG = RagConfig.model_validate(
    {"chunking": {"strategy": "recursive", "chunk_size": 128, "overlap": 16}}
).model_dump(mode="json")


def _record(run_id: str, mrr: float, latency: float, chunk_size: int) -> dict:
    config = json.loads(json.dumps(CONFIG))
    config["chunking"]["chunk_size"] = chunk_size
    return {
        "run_id": run_id,
        "config": config,
        "metrics": {"mrr": mrr, "hit_at_k": 1.0},
        "latency_ms": {"mean_ms": latency, "median_ms": latency},
    }


def _write_results(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


def test_load_experiment_records_includes_config_and_latency(tmp_path: Path) -> None:
    _write_results(tmp_path, [_record("run-0", 0.9, 12.5, 256)])

    # Accepts the output directory as well as the file itself.
    records = load_experiment_records(tmp_path)

    assert records[0]["run_id"] == "run-0"
    assert records[0]["config"]["chunking"]["chunk_size"] == 256
    assert records[0]["metrics"] == {"mrr": 0.9, "hit_at_k": 1.0, "latency_ms": 12.5}


def test_load_experiment_records_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rag experiment"):
        load_experiment_records(tmp_path)


def test_load_experiment_records_empty(tmp_path: Path) -> None:
    _write_results(tmp_path, [])
    with pytest.raises(ValueError, match="no runs"):
        load_experiment_records(tmp_path)


def test_optimizer_returns_full_config_of_best_run(tmp_path: Path) -> None:
    path = _write_results(
        tmp_path,
        [_record("run-0", 0.5, 5.0, 128), _record("run-1", 0.9, 5.0, 512)],
    )

    result = ParetoOptimizer().optimize(load_experiment_records(path), {"primary_metric": "mrr"})

    assert result.recommended_run_id == "run-1"
    assert result.recommended_config["chunking"]["chunk_size"] == 512
    RagConfig.model_validate(result.recommended_config)
    assert "run-1" in result.reasoning


def test_optimizer_minimizes_latency_and_honours_constraints(tmp_path: Path) -> None:
    path = _write_results(
        tmp_path,
        [
            _record("slow", 0.9, 100.0, 512),
            _record("fast", 0.9, 5.0, 256),
            _record("fast-worse", 0.7, 4.0, 128),
        ],
    )
    records = load_experiment_records(path)

    result = ParetoOptimizer().optimize(records, {"primary_metric": "mrr"})
    frontier = {point.run_id for point in result.pareto_frontier}
    assert "slow" not in frontier  # same quality, higher latency: dominated
    assert result.recommended_run_id == "fast"

    constrained = ParetoOptimizer().optimize(
        records, {"primary_metric": "mrr", "constraints": {"latency_ms": 4.5}}
    )
    assert constrained.recommended_run_id == "fast-worse"


def test_optimizer_baseline_comparison_by_run_id(tmp_path: Path) -> None:
    path = _write_results(
        tmp_path,
        [_record("run-0", 0.5, 5.0, 128), _record("run-1", 0.9, 5.0, 512)],
    )

    result = ParetoOptimizer().optimize(
        load_experiment_records(path), {"primary_metric": "mrr", "baseline_run_id": "run-0"}
    )

    assert result.baseline_comparison is not None
    assert result.baseline_comparison["mrr"] == pytest.approx(0.4)


def test_reasoning_handles_missing_metric(tmp_path: Path) -> None:
    path = _write_results(tmp_path, [_record("run-0", 0.5, 5.0, 128)])

    result = ParetoOptimizer().optimize(
        load_experiment_records(path),
        {"primary_metric": "mrr", "secondary_metric": "ndcg_at_k"},
    )

    assert "ndcg_at_k = N/A" in result.reasoning
