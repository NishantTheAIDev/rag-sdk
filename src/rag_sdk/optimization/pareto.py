"""Pareto optimizer implementation."""

from __future__ import annotations

from typing import Any

from rag_sdk.optimization.base import OptimizationResult, Optimizer, ParetoPoint, is_pareto_optimal


class ParetoOptimizer(Optimizer):
    """Pareto optimizer for multi-objective optimization."""

    def optimize(
        self,
        experiment_results: list[dict[str, Any]],
        optimization_config: dict[str, Any],
    ) -> OptimizationResult:
        """Find Pareto-optimal configurations and recommend the best one."""
        if not experiment_results:
            raise ValueError("No experiment results to optimize")

        # Extract metrics from results. ``run_id`` may sit on the record or,
        # for older callers, inside ``config``.
        configs_with_metrics = []
        run_ids: dict[int, str | None] = {}
        for result in experiment_results:
            config = result.get("config", {})
            metrics = result.get("metrics", {})
            if metrics:
                run_ids[id(config)] = result.get("run_id") or config.get("run_id")
                configs_with_metrics.append((config, metrics))

        if not configs_with_metrics:
            raise ValueError("No valid metrics in experiment results")

        # Determine which metrics to maximize/minimize
        primary_metric = optimization_config.get("primary_metric", "mrr")
        secondary_metric = optimization_config.get("secondary_metric")
        constraints = optimization_config.get("constraints", {})
        weights = optimization_config.get("weights", {})

        # Default: maximize retrieval metrics, minimize latency
        maximize_metrics = {
            "hit_at_k",
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "ndcg_at_k",
            "map",
            "faithfulness",
            "answer_relevance",
        }
        minimize_metrics = {"latency_ms"}

        # Filter by constraints
        valid_configs = []
        for config, metrics in configs_with_metrics:
            satisfies = True
            for constraint_key, constraint_value in constraints.items():
                if constraint_key in metrics:
                    if constraint_key in minimize_metrics:
                        if metrics[constraint_key] > constraint_value:
                            satisfies = False
                            break
                    else:
                        if metrics[constraint_key] < constraint_value:
                            satisfies = False
                            break
            if satisfies:
                valid_configs.append((config, metrics))

        if not valid_configs:
            # Fall back to all configs if none satisfy constraints
            valid_configs = configs_with_metrics

        # Find Pareto frontier
        all_metrics = [m for _, m in valid_configs]
        pareto_points = []

        for config, metrics in valid_configs:
            if is_pareto_optimal(metrics, all_metrics, maximize_metrics, minimize_metrics):
                pareto_points.append(
                    ParetoPoint(
                        config=config,
                        metrics=metrics,
                        dominated=False,
                        run_id=run_ids.get(id(config)),
                    )
                )

        # Score each Pareto point
        scored_points = []
        for point in pareto_points:
            score = self._score_point(point.metrics, primary_metric, secondary_metric, weights)
            scored_points.append((score, point))

        # Sort by score (higher is better)
        scored_points.sort(key=lambda x: x[0], reverse=True)

        best_point = scored_points[0][1] if scored_points else ParetoPoint(config={}, metrics={})

        # Generate reasoning
        reasoning = self._generate_reasoning(
            best_point,
            scored_points,
            primary_metric,
            secondary_metric,
            constraints,
        )

        # Baseline comparison
        baseline_comparison = None
        baseline_run_id = optimization_config.get("baseline_run_id")
        if baseline_run_id:
            baseline_metrics = next(
                (
                    m
                    for c, m in configs_with_metrics
                    if run_ids.get(id(c)) == baseline_run_id
                ),
                None,
            )
            if baseline_metrics is not None:
                baseline_comparison = {}
                for key in set(best_point.metrics.keys()) | set(baseline_metrics.keys()):
                    if key in best_point.metrics and key in baseline_metrics:
                        baseline_comparison[key] = best_point.metrics[key] - baseline_metrics[key]

        return OptimizationResult(
            recommended_config=best_point.config,
            recommended_run_id=best_point.run_id,
            reasoning=reasoning,
            pareto_frontier=pareto_points,
            baseline_comparison=baseline_comparison,
            all_configs=[c for c, _ in configs_with_metrics],
        )

    def _score_point(
        self,
        metrics: dict[str, float],
        primary_metric: str,
        secondary_metric: str | None,
        weights: dict[str, float],
    ) -> float:
        """Score a Pareto point."""
        score = 0.0

        # Primary metric weight
        primary_weight = weights.get(primary_metric, 1.0)
        score += metrics.get(primary_metric, 0) * primary_weight

        # Secondary metric
        if secondary_metric:
            secondary_weight = weights.get(secondary_metric, 0.5)
            score += metrics.get(secondary_metric, 0) * secondary_weight

        # Other weights
        for key, weight in weights.items():
            if key not in (primary_metric, secondary_metric):
                score += metrics.get(key, 0) * weight

        return score

    def _generate_reasoning(
        self,
        best_point: ParetoPoint,
        all_scored: list[tuple[float, ParetoPoint]],
        primary_metric: str,
        secondary_metric: str | None,
        constraints: dict[str, float],
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        def fmt(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.4f}"

        lines = [
            f"Recommended configuration selected from {len(all_scored)} "
            f"Pareto-optimal candidates.",
        ]
        if best_point.run_id:
            lines.append(f"Recommended run: {best_point.run_id}")
        lines.append(
            f"Primary metric: {primary_metric} = {fmt(best_point.metrics.get(primary_metric))}"
        )

        if secondary_metric:
            val = best_point.metrics.get(secondary_metric)
            lines.append(f"Secondary metric: {secondary_metric} = {fmt(val)}")

        if constraints:
            lines.append("Constraints satisfied:")
            for key, value in constraints.items():
                actual = best_point.metrics.get(key)
                lines.append(f"  {key}: {fmt(actual)} (limit: {value})")

        lines.append(f"\nPareto frontier size: {len(all_scored)}")
        lines.append("Top 3 candidates:")
        for i, (score, point) in enumerate(all_scored[:3], 1):
            val = point.metrics.get(primary_metric)
            label = f"{point.run_id} | " if point.run_id else ""
            lines.append(f"  {i}. {label}Score: {score:.4f} | {primary_metric}: {fmt(val)}")

        return "\n".join(lines)