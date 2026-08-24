"""Optimization base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""

    config: dict[str, Any]
    metrics: dict[str, float]
    dominated: bool = False


class OptimizationResult(BaseModel):
    """Result of optimization."""

    model_config = ConfigDict(extra="forbid")

    recommended_config: dict[str, Any]
    reasoning: str
    pareto_frontier: list[ParetoPoint]
    baseline_comparison: dict[str, float] | None = None
    all_configs: list[dict[str, Any]] = []


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def optimize(
        self,
        experiment_results: list[dict[str, Any]],
        optimization_config: dict[str, Any],
    ) -> OptimizationResult:
        """Find the optimal configuration from experiment results."""
        ...


def is_pareto_optimal(
    metrics: dict[str, float],
    other_points: list[dict[str, float]],
    maximize: list[str],
    minimize: list[str],
) -> bool:
    """Check if a point is Pareto optimal."""
    for other in other_points:
        dominates = True
        strictly_better = False

        for key in maximize:
            if other.get(key, 0) < metrics.get(key, 0):
                dominates = False
                break
            if other.get(key, 0) > metrics.get(key, 0):
                strictly_better = True

        for key in minimize:
            if other.get(key, 0) > metrics.get(key, 0):
                dominates = False
                break
            if other.get(key, 0) < metrics.get(key, 0):
                strictly_better = True

        if dominates and strictly_better:
            return False

    return True