"""Experiment engine."""

from __future__ import annotations

from rag_sdk.dataset import QuerySample
from rag_sdk.experiments.dataset import DatasetError, load_queries
from rag_sdk.experiments.grid import (
    ExperimentParameterWarning,
    GridVariant,
    apply_override,
    expand_grid,
    expand_grid_with_detail,
)
from rag_sdk.experiments.records import (
    EmbeddingInfo,
    ExperimentRecord,
    ExperimentResult,
    LatencyStats,
)
from rag_sdk.experiments.reports import (
    write_csv,
    write_html,
    write_json,
    write_leaderboard,
    write_reports,
)
from rag_sdk.experiments.runner import ExperimentRunner, run_experiment

__all__ = [
    "DatasetError",
    "EmbeddingInfo",
    "ExperimentParameterWarning",
    "ExperimentRecord",
    "ExperimentResult",
    "ExperimentRunner",
    "GridVariant",
    "LatencyStats",
    "QuerySample",
    "apply_override",
    "expand_grid",
    "expand_grid_with_detail",
    "load_queries",
    "run_experiment",
    "write_csv",
    "write_html",
    "write_json",
    "write_leaderboard",
    "write_reports",
]