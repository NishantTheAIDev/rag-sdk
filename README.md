# RAG SDK

Configuration-driven RAG experimentation, evaluation, and deployment.

A Python SDK for building, experimenting with, evaluating, and deploying
production-grade RAG pipelines by editing a YAML config instead of writing
boilerplate. Compare chunking, embedding, retrieval, reranking, and evaluation
strategies and let the SDK measure what works best for your data.

## Features

- Typed YAML configuration validated with Pydantic v2
- Pluggable chunking: recursive and fixed-token strategies with overlap
- `EmbeddingProvider` interface for swappable embedding backends
- FAISS-backed vector store (`FaissVectorStore`)
- Dense, BM25, and hybrid (RRF/weighted fusion) retrieval
- Retrieval metrics: Hit@K, Recall@K, Precision@K, MRR, nDCG, MAP
- Configuration-driven experiment engine with parameter sweeps
- CSV, JSON, leaderboard, and interactive HTML experiment reports
- Thin `rag` CLI (`init`, `validate`, `evaluate`, `experiment`)

## Install

```bash
uv pip install -e ".[dev]"
```

## Quickstart

```bash
rag init        # write rag.yaml
rag validate rag.yaml
```

```python
from rag_sdk.chunking import build_chunker
from rag_sdk.config import load_config
from rag_sdk.core import Document
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import DenseRetriever

config = load_config("rag.yaml")
chunker = build_chunker(config.chunking)
chunks = chunker.chunk(Document(id="d1", text="..."))

store = FaissVectorStore(dimension=128)
retriever = DenseRetriever(embedding_provider, store)
retriever.add_chunks(chunks)
results = retriever.search("query", top_k=config.retrieval.top_k)
```

## Running experiments

Declare parameter sweeps in `rag.yaml` and run them from the CLI:

```yaml
documents:
  path: ./docs

experiments:
  dataset: ./queries.jsonl
  parameters:
    chunking.chunk_size: [256, 512]
    retrieval.strategy: [dense, hybrid]
```

```bash
rag experiment rag.yaml
```

See `configs/experiment.yaml` for a complete example.

## Documentation

See `docs/` for the product specification and API reference. Build the docs
with:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

## Development

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
```
