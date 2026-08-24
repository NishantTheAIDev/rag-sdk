"""RAG SDK command line interface.

The CLI stays thin: it reads inputs, delegates to SDK modules, and prints
results. All business logic lives in the SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_sdk.config import (
    ConfigError,
    baseline_config,
    default_config,
    dump_config,
    load_config,
)
from rag_sdk.evaluation import evaluate_rag, evaluate_retrieval
from rag_sdk.evaluation.io import load_retrieval_results
from rag_sdk.experiments import run_experiment, write_reports
from rag_sdk.ingestion import IngestionError, load_documents
from rag_sdk.optimization import build_optimizer

app = typer.Typer(
    name="rag",
    help="Configuration-driven RAG experimentation and evaluation.",
    no_args_is_help=True,
)


@app.command()
def init(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to write the configuration to."),
    ] = Path("rag.yaml"),
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite the output file if it exists.")
    ] = False,
) -> None:
    """Write a starter configuration to a YAML file."""
    if output.exists() and not force:
        raise typer.BadParameter(f"{output} already exists; use --force to overwrite")
    output.write_text(dump_config(default_config()), encoding="utf-8")
    typer.echo(f"Wrote configuration to {output}")


@app.command()
def validate(config: Path) -> None:
    """Validate a configuration file and print a summary."""
    try:
        loaded = load_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Valid configuration: project={loaded.project.name!r}, "
        f"chunking={loaded.chunking.strategy!r}, top_k={loaded.retrieval.top_k}"
    )


@app.command()
def evaluate(
    results_file: Path,
    k: Annotated[int, typer.Option("--k", min=1, help="Rank cutoff for the metrics.")] = 10,
) -> None:
    """Compute retrieval metrics from a JSONL results file."""
    try:
        samples = load_retrieval_results(results_file)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not samples:
        raise typer.BadParameter("results file contains no samples")
    metrics = evaluate_retrieval(samples, k=k)
    typer.echo(f"{'metric':<16}{'value':>10}")
    for name, value in metrics.items():
        typer.echo(f"{name:<16}{value:>10.4f}")


@app.command()
def benchmark(
    documents: Annotated[Path, typer.Argument(help="Path to document corpus.")],
    dataset: Annotated[Path, typer.Argument(help="Path to evaluation dataset (JSONL).")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory to write results."),
    ] = None,
    baseline_version: Annotated[
        str, typer.Option("--version", help="Baseline version to use.")
    ] = "v1",
) -> None:
    """Run the baseline configuration on a corpus and dataset."""
    try:
        docs = load_documents(documents)
    except IngestionError as exc:
        raise typer.BadParameter(str(exc)) from exc

    config = baseline_config(baseline_version)
    config.documents = None  # We pass documents directly

    try:
        result = evaluate_rag(config, docs, str(dataset))
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    metrics = result["metrics"]
    typer.echo(f"Baseline {baseline_version} results:")
    typer.echo(f"{'metric':<24}{'value':>10}")
    for name, value in metrics.items():
        typer.echo(f"{name:<24}{value:>10.4f}")

    if output:
        output.mkdir(parents=True, exist_ok=True)
        import json
        (output / "baseline_results.json").write_text(
            json.dumps({"metrics": metrics, "config": config.model_dump(mode="json")}, indent=2)
        )
        typer.echo(f"Wrote results to {output / 'baseline_results.json'}")


@app.command()
def experiment(
    config: Path,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory to write reports into."),
    ] = None,
) -> None:
    """Run a parameter sweep experiment and write reports."""
    try:
        loaded = load_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if loaded.experiments is None:
        raise typer.BadParameter(
            f"{config} has no 'experiments' section; see configs/experiment.yaml"
        )
    if loaded.documents is None:
        raise typer.BadParameter(
            f"{config} has no 'documents.path' section pointing at a corpus"
        )
    try:
        documents = load_documents(loaded.documents.path)
    except IngestionError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        result = run_experiment(documents, loaded)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_dir = Path(output or loaded.experiments.output_dir)
    paths = write_reports(result, output_dir)

    recommendation = result.leaderboard()[0]
    typer.echo(
        f"Ran {len(result.records)} configuration(s) across "
        f"{len(documents)} document(s)."
    )
    typer.echo(
        f"Best by {result.primary_metric}: {recommendation.run_id} "
        f"({recommendation.metrics[result.primary_metric]:.4f})"
    )
    typer.echo(f"Wrote reports to {output_dir}")
    for name, path in paths.items():
        typer.echo(f"  {name:<12} {path}")


@app.command()
def optimize(
    config: Path,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory with experiment results."),
    ] = None,
) -> None:
    """Run optimization on experiment results and print recommendation."""
    try:
        loaded = load_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if loaded.experiments is None:
        raise typer.BadParameter(f"{config} has no 'experiments' section")

    exp_output_dir = Path(output or loaded.experiments.output_dir)
    leaderboard_path = exp_output_dir / "leaderboard.csv"

    if not leaderboard_path.exists():
        raise typer.BadParameter(f"Leaderboard not found at {leaderboard_path}")

    import csv
    with leaderboard_path.open() as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            metrics = {k: float(v) for k, v in row.items() if k not in ("run_id", "config")}
            config_data = {"run_id": row["run_id"]}
            try:
                import json
                config_data["config"] = json.loads(row.get("config", "{}"))
            except (json.JSONDecodeError, KeyError):
                pass
            records.append({"config": config_data, "metrics": metrics})

    optimizer = build_optimizer("pareto")
    opt_config = loaded.optimization.model_dump() if loaded.optimization else {}
    opt_result = optimizer.optimize(records, opt_config)

    typer.echo("Optimization Result")
    typer.echo("=" * 50)
    typer.echo(opt_result.reasoning)
    typer.echo()
    typer.echo("Recommended Configuration:")
    import yaml
    typer.echo(yaml.dump(opt_result.recommended_config, sort_keys=False))

    if opt_result.baseline_comparison:
        typer.echo("Baseline Comparison:")
        for key, diff in opt_result.baseline_comparison.items():
            sign = "+" if diff >= 0 else ""
            typer.echo(f"  {key}: {sign}{diff:.4f}")

    if output:
        out_path = Path(output) / "optimized-rag.yaml"
        out_path.write_text(yaml.dump(opt_result.recommended_config, sort_keys=False))
        typer.echo(f"Wrote optimized config to {out_path}")


@app.command()
def export_config(
    config: Path,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path."),
    ] = None,
    format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (yaml or json).")
    ] = "yaml",
) -> None:
    """Export a configuration as YAML or JSON."""
    try:
        loaded = load_config(config)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    out_path = output or Path(f"optimized-rag.{format}")
    if format == "yaml":
        out_path.write_text(dump_config(loaded), encoding="utf-8")
    elif format == "json":
        import json
        out_path.write_text(json.dumps(loaded.model_dump(mode="json", exclude_none=True), indent=2))
    else:
        raise typer.BadParameter(f"Unknown format: {format}")

    typer.echo(f"Exported configuration to {out_path}")