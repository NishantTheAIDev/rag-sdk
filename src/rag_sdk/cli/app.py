"""RAG SDK command line interface.

The CLI stays thin: it reads inputs, delegates to SDK modules, and prints
results. All business logic lives in the SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_sdk.config import ConfigError, default_config, dump_config, load_config
from rag_sdk.evaluation import evaluate_retrieval
from rag_sdk.evaluation.io import load_retrieval_results
from rag_sdk.experiments import run_experiment, write_reports
from rag_sdk.ingestion import IngestionError, load_documents

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