# RAG SDK

Configuration-driven RAG experimentation, evaluation, and deployment.

A Python SDK for building, experimenting with, evaluating, and deploying
production-grade RAG pipelines by editing a YAML config instead of writing
boilerplate. Compare chunking, embedding, retrieval, reranking, and evaluation
strategies and let the SDK measure what works best for your data.

## Features

- Typed YAML configuration validated with Pydantic v2
- Pluggable chunking: recursive, fixed-token, **sentence window**, **parent-child**
- `EmbeddingProvider` interface for swappable embedding backends
- FAISS-backed vector store (`FaissVectorStore`)
- Dense, BM25, and hybrid (RRF/weighted fusion) retrieval
- **Reranking**: cross-encoder, Cohere v4, or none (baseline)
- **Context enrichment**: parent-child expansion, sentence window, auto-merging
- Retrieval metrics: Hit@K, Recall@K, Precision@K, MRR, nDCG, MAP
- Configuration-driven experiment engine with parameter sweeps
- CSV, JSON, leaderboard, and interactive HTML experiment reports
- Thin `rag` CLI (`init`, `validate`, `evaluate`, `experiment`)

## Install

```bash
uv pip install -e ".[dev]"
# For reranking: uv pip install -e ".[reranker]"
# For all:       uv pip install -e ".[all]"
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

## Configuration Examples

### 1. Basic Dense Retrieval

```yaml
project:
  name: basic-rag

documents:
  path: ./data/docs

chunking:
  strategy: recursive
  chunk_size: 512
  overlap: 64

embedding:
  provider: sentence-transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: dense
  top_k: 5

experiments:
  dataset: ./data/queries.jsonl
  k: 10
  primary_metric: mrr
  output_dir: ./runs
  parameters:
    chunking.chunk_size: [256, 512, 768]
```

### 2. Hybrid Retrieval (BM25 + Dense)

```yaml
project:
  name: hybrid-rag

documents:
  path: ./data/docs

chunking:
  strategy: recursive
  chunk_size: 512
  overlap: 64

embedding:
  provider: sentence-transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: hybrid
  top_k: 5
  candidate_k: 50
  fusion:
    method: rrf
    rrf_k: 60
  bm25:
    k1: 1.5
    b: 0.75

experiments:
  dataset: ./data/queries.jsonl
  k: 10
  primary_metric: mrr
  output_dir: ./runs
  parameters:
    chunking.chunk_size: [256, 512]
    retrieval.strategy: [dense, hybrid, bm25]
    retrieval.fusion.method: [rrf, weighted]
```

### 3. Parent-Child Retrieval with Reranking

```yaml
project:
  name: parent-child-rag

documents:
  path: ./data/docs

chunking:
  strategy: parent_child
  parent_chunk_size: 1024
  parent_overlap: 128
  child_chunk_size: 256
  child_overlap: 32

embedding:
  provider: sentence-transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: hybrid
  top_k: 5                    # Final output after enrichment
  candidate_k: 50             # Retriever pool before reranking
  fusion:
    method: rrf
  parent_child:
    enabled: true             # Expand child hits to parent chunks

reranker:
  strategy: cross_encoder
  model: BAAI/bge-reranker-base
  top_k: 5                    # Output after reranking

experiments:
  dataset: ./data/queries.jsonl
  k: 10
  primary_metric: mrr
  output_dir: ./runs
  parameters:
    chunking.strategy: [recursive, parent_child]
    retrieval.strategy: [dense, hybrid]
    reranker.strategy: [none, cross_encoder]  # baseline vs reranked
    retrieval.parent_child.enabled: [true, false]
```

### 4. Sentence Window Expansion

```yaml
project:
  name: sentence-window-rag

documents:
  path: ./data/docs

chunking:
  strategy: recursive
  chunk_size: 512
  overlap: 64

embedding:
  provider: sentence-transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: hybrid
  top_k: 5
  candidate_k: 50
  fusion:
    method: rrf
  sentence_window:
    enabled: true
    window_size: 3            # Expand to 3 sentences around match

experiments:
  dataset: ./data/queries.jsonl
  k: 10
  primary_metric: mrr
  output_dir: ./runs
  parameters:
    retrieval.sentence_window.enabled: [true, false]
    retrieval.sentence_window.window_size: [3, 5]
```

### 5. Auto-Merging

```yaml
project:
  name: auto-merging-rag

documents:
  path: ./data/docs

chunking:
  strategy: recursive
  chunk_size: 256
  overlap: 32

embedding:
  provider: sentence-transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: hybrid
  top_k: 5
  candidate_k: 50
  auto_merging:
    enabled: true
    similarity_threshold: 0.8   # Merge if cosine similarity >= 0.8
    max_tokens: 512             # Token limit for merged chunk
    tokenizer: whitespace

experiments:
  dataset: ./data/queries.jsonl
  k: 10
  primary_metric: mrr
  output_dir: ./runs
  parameters:
    retrieval.auto_merging.enabled: [true, false]
    retrieval.auto_merging.similarity_threshold: [0.7, 0.8, 0.9]
```

### 6. Full Configuration: All Enrichment + Reranking

```yaml
project:
  name: full-rag

documents:
  path: ./data/docs

chunking:
  strategy: parent_child
  parent_chunk_size: 1024
  parent_overlap: 128
  child_chunk_size: 256
  child_overlap: 32

embedding:
  provider: sentence-transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: hybrid
  top_k: 5                    # Final output
  candidate_k: 50             # Before reranking
  fusion:
    method: rrf
  # Explicit enrichment (not inferred from chunker)
  sentence_window:
    enabled: true
    window_size: 3
  parent_child:
    enabled: true
  auto_merging:
    enabled: true
    similarity_threshold: 0.8
    max_tokens: 512
    tokenizer: whitespace

reranker:
  strategy: cross_encoder
  model: BAAI/bge-reranker-base
  top_k: 5                    # Post-rerank pool

experiments:
  dataset: ./data/queries.jsonl
  k: 10
  primary_metric: mrr
  output_dir: ./runs
  parameters:
    chunking.strategy: [recursive, parent_child]
    retrieval.strategy: [dense, hybrid]
    reranker.strategy: [none, cross_encoder, cohere]
    retrieval.sentence_window.enabled: [true, false]
    retrieval.parent_child.enabled: [true, false]
    retrieval.auto_merging.enabled: [true, false]
```

### 7. Using Cohere Reranker (requires API key)

```yaml
reranker:
  strategy: cohere
  model: rerank-v4.0-fast
  api_key: ${COHERE_API_KEY}  # or set COHERE_API_KEY env var
  top_k: 5
```

Install with: `uv pip install -e ".[reranker]"`

## Pipeline Architecture

```
Query
  ↓
[Retriever] ──→ candidate_k results (e.g., 50)
  ↓
[Reranker]  ──→ reranker.top_k results (e.g., 5)
  ↓
[Enrichment] ──→ final retrieval.top_k (e.g., 5)
     ├─ Parent-Child: map child → parent via ChunkStore
     ├─ Sentence Window: expand via DocumentStore + boundaries
     └─ Auto-Merge: merge adjacent similar chunks via stored embeddings
```

Key points:
- `candidate_k` = retriever pool size (before reranking)
- `reranker.top_k` = post-rerank pool size
- `retrieval.top_k` = final output after enrichment
- Lineage preserved: `source_chunk_id`, `child_ids`, `merged_source_ids`

## Documentation

See `docs/` for the product specification and API reference. Build the docs with:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

## Development

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
```