# RAG SDK — Agent Instructions

## Project

This repository contains an open-source Python SDK for building,
experimenting with, evaluating, and deploying production-grade RAG
pipelines.

The primary differentiator is configuration-driven RAG experimentation:
developers should be able to compare chunking, embedding, retrieval,
reranking, and evaluation strategies without writing custom boilerplate.

Read `docs/product-spec.md` for the complete product vision and
requirements.

## Architecture Principles

- Configuration-first
- Modular and pluggable
- Framework agnostic
- Strong typing
- Production quality
- Testable
- Extensible
- Backwards-compatible public APIs

Prefer interfaces/abstract base classes for replaceable RAG components.

Examples:

- Chunker
- EmbeddingProvider
- VectorStore
- Retriever
- Reranker
- Evaluator
- ExperimentRunner

Do not tightly couple these components.

## Technology

- Python 3.14+
- Pydantic v2
- Typer
- Pytest
- Ruff
- MkDocs
- Pyproject-based packaging

Use type hints throughout the codebase.

## Development Environment

This project uses `uv` for dependency management. Use the project's `.venv` for all commands:

```bash
# Install dependencies
uv sync --all-extras

# Run commands in the venv
uv run pytest
uv run ruff check .
uv run mkdocs serve
```

Do not use `pip install` directly; the `.venv` is managed by `uv`.

## Development Rules

Before implementing a major feature:

1. Inspect the existing architecture.
2. Identify affected interfaces.
3. Propose the design.
4. Implement incrementally.
5. Add tests.
6. Update documentation.

Do not rewrite large portions of the repository unnecessarily.

Do not introduce a dependency unless there is a clear reason.

Keep provider-specific implementations behind adapters.

## Experimentation

Experiments must be reproducible.

Configurations should be serializable.

Experiment results should record:

- configuration
- dataset/version
- metrics
- latency
- timestamp
- model information

Do not hardcode experiment parameters in Python.

## Evaluation

Keep retrieval evaluation separate from answer-generation evaluation.

Retrieval metrics include:

- Hit@K
- Recall@K
- Precision@K
- MRR
- nDCG
- MAP

Answer evaluation should be independently extensible.

## Public API

Treat anything exposed from the top-level package as public API.

Avoid unnecessary breaking changes.

Prefer explicit APIs over magic behavior.

## CLI

The CLI should remain thin.

Business logic belongs in the SDK modules, not inside CLI commands.

## Testing

Every new component should have unit tests.

Integration tests should cover important end-to-end flows.

Tests should not require external API keys unless explicitly marked
as integration tests.

## Current Development Strategy

Build incrementally:

1. Core interfaces and configuration
2. Chunking
3. Embeddings
4. Indexing
5. Dense retrieval
6. BM25/hybrid retrieval
7. Evaluation
8. Experiment engine
9. Advanced retrieval
10. Reranking
11. Answer evaluation
12. Observability
13. Production runtime

Do not jump directly to the complete production implementation.