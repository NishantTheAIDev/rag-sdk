"""Structured experiment records."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class LatencyStats(BaseModel):
    """Search latency in milliseconds across dataset queries."""

    model_config = ConfigDict(frozen=True)

    mean_ms: float
    median_ms: float


class EmbeddingInfo(BaseModel):
    """Which embedding provider served an experiment run."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str | None
    dimension: int


class ExperimentRecord(BaseModel):
    """Everything captured for a single experiment run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    config: dict[str, object]
    dataset_path: str
    dataset_hash: str
    timestamp: datetime
    embedding: EmbeddingInfo
    latency_ms: LatencyStats
    metrics: dict[str, float] = Field(default_factory=dict)
    total_chunks: int
    skipped_parameters: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """All records produced by an experiment."""

    model_config = ConfigDict(frozen=True)

    dataset_path: str
    dataset_hash: str
    primary_metric: str
    k: int
    relevance_level: str = "document"
    records: list[ExperimentRecord]

    def leaderboard(self) -> list[ExperimentRecord]:
        """Records sorted best-first by the primary metric."""
        return sorted(
            self.records,
            key=lambda record: record.metrics[self.primary_metric],
            reverse=True,
        )