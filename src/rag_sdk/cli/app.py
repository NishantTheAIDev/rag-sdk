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