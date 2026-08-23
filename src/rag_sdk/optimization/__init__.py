"""Optimization engine for RAG configuration."""

from __future__ import annotations

from rag_sdk.optimization.base import OptimizationResult, Optimizer, ParetoPoint
from rag_sdk.optimization.factory import build_optimizer
from rag_sdk.optimization.pareto import ParetoOptimizer

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "ParetoPoint",
    "ParetoOptimizer",
    "build_optimizer",
]