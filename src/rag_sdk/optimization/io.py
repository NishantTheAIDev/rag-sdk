"""Loading experiment results as optimizer input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_FILENAME = "results.json"


def load_experiment_records(path: str | Path) -> list[dict[str, Any]]:
    """Load experiment runs from ``results.json`` (or a directory holding it).

    Each returned record is ``{"run_id", "config", "metrics"}`` where
    ``config`` is the full pipeline configuration of the run and ``metrics``
    holds the retrieval metrics plus ``latency_ms`` (mean search latency), so
    latency can be minimized or constrained during optimization.
    """
    results_path = Path(path)
    if results_path.is_dir():
        results_path = results_path / RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(
            f"Experiment results not found at {results_path}; run `rag experiment` first"
        )
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid experiment results file {results_path}: {exc}") from exc

    records: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        metrics = {name: float(value) for name, value in record.get("metrics", {}).items()}
        latency = record.get("latency_ms") or {}
        if "mean_ms" in latency:
            metrics["latency_ms"] = float(latency["mean_ms"])
        records.append(
            {
                "run_id": record.get("run_id"),
                "config": record.get("config", {}),
                "metrics": metrics,
            }
        )
    if not records:
        raise ValueError(f"Experiment results file {results_path} contains no runs")
    return records
