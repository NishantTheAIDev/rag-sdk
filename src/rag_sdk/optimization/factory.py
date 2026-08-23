"""Optimizer factory."""

from __future__ import annotations

from rag_sdk.optimization import Optimizer
from rag_sdk.optimization.pareto import ParetoOptimizer


def build_optimizer(optimizer_type: str = "pareto") -> Optimizer:
    """Construct an optimizer from its type."""
    if optimizer_type == "pareto":
        return ParetoOptimizer()
    raise ValueError(f"Unknown optimizer type: {optimizer_type!r}")