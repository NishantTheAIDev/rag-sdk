# RAG SDK

A Python SDK for building, experimenting with, evaluating, and deploying
production-grade RAG pipelines through configuration instead of boilerplate.

The primary goal is to answer **"which RAG configuration works best for my
data?"** by making chunking, embedding, retrieval, reranking, and evaluation
configurable and measurable.

## Quickstart

Install the package:

```bash
uv pip install -e ".[dev]"
```

Create a starter configuration:

```bash
rag init
```

```yaml
# rag.yaml
project:
  name: api-rag

chunking:
  strategy: recursive
  chunk_size: 512
  overlap: 64

embedding:
  provider: hash

retrieval:
  strategy: dense
  top_k: 5
```

Validate it:

```bash
rag validate rag.yaml
```

## Using the SDK

```python
from rag_sdk.chunking import build_chunker
from rag_sdk.config import load_config
from rag_sdk.core import Document
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import DenseRetriever

config = load_config("rag.yaml")
chunker = build_chunker(config.chunking)

doc = Document(id="d1", text="Cats are small carnivorous mammals.")
chunks = chunker.chunk(doc)

store = FaissVectorStore(dimension=128)
retriever = DenseRetriever(embedding_provider, store)  # see EmbeddingProvider
retriever.add_chunks(chunks)

results = retriever.search("kittens", top_k=config.retrieval.top_k)
```

## Evaluating retrieval

```python
from rag_sdk.evaluation import evaluate_retrieval

samples = [
    (["c1", "c2", "c3"], {"c1"}),
    (["c4", "c5", "c6"], {"c5", "c6"}),
]
metrics = evaluate_retrieval(samples, k=3)
print(metrics)
```

Or from the CLI:

```bash
rag evaluate results.jsonl --k 10
```

where each line of `results.jsonl` is
`{"retrieved": ["c1", "c2"], "relevant": ["c1"]}`.

## Configuration reference

| Section      | Field        | Default   | Description                     |
| ------------ | ------------ | --------- | ------------------------------- |
| `project`    | `name`       | `rag-project` | Project name                |
| `chunking`   | `strategy`   | `recursive` | Chunking strategy            |
| `chunking`   | `chunk_size` | `512`     | Target size per chunk            |
| `chunking`   | `overlap`    | `64`      | Overlap between adjacent chunks  |
| `embedding`  | `provider`   | `hash`    | Embedding provider name          |
| `retrieval`  | `strategy`   | `dense`   | Retrieval strategy               |
| `retrieval`  | `top_k`      | `5`       | Number of results returned       |

`chunking.strategy` supports `recursive` and `fixed`. Each strategy validates
its own fields; unknown strategies or extra keys are rejected.

## Development

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
```
