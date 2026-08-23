# RAG SDK

A Python SDK for building, experimenting with, evaluating, and optimizing
production-grade RAG pipelines through configuration instead of boilerplate.

The primary goal is to answer **"which RAG configuration works best for my
data?"** by making chunking, embedding, retrieval, reranking, generation,
evaluation, and optimization configurable and measurable.

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

## Hybrid retrieval

Retrieval is config-driven. `strategy` supports `dense`, `bm25`, `hybrid`, and `mmr`:

```yaml
retrieval:
  strategy: hybrid
  top_k: 5
  fusion:
    method: rrf          # rrf | weighted
    candidate_k: 20      # candidates gathered from each retriever
  bm25:
    k1: 1.5
    b: 0.75
```

MMR (Maximal Marginal Relevance):

```yaml
retrieval:
  strategy: mmr
  top_k: 10
  candidate_k: 50
  lambda_param: 0.5      # 0 = diversity, 1 = relevance
```

## Experiments

The `experiments:` section declares parameter sweeps as dot-paths into the
pipeline config. Every combination is run and evaluated against a JSONL
dataset where each line is
`{"query": "...", "relevant_documents": ["doc_id", ...]}`:

```yaml
documents:
  path: ./docs

experiments:
  dataset: ./queries.jsonl
  k: 10
  primary_metric: mrr      # hit_at_k | precision_at_k | recall_at_k | mrr | ndcg_at_k | map
  output_dir: ./runs
  parameters:
    chunking.chunk_size: [256, 512]
    retrieval.strategy: [dense, hybrid]
```

Run it:

```bash
rag experiment rag.yaml
```

A parameter that does not apply to a strategy (for example
`retrieval.fusion.method` on a `dense` combination) is skipped with a warning
and recorded in that run's `skipped_parameters` metadata.

Reports are written to `output_dir`:

- `results.csv` — one row per configuration
- `results.json` — full records plus a leaderboard
- `leaderboard.csv` — configurations ranked by the primary metric
- `report.html` — interactive sortable report with a recommended configuration

## End-to-End RAG with Generation & Answer Evaluation

The SDK supports full RAG evaluation including answer generation and quality assessment.

### Generation

```yaml
generation:
  provider: openai      # mock | openai | anthropic | ollama
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 512
```

### Answer Evaluation

```yaml
evaluation:
  answer:
    enabled: true
    reference_based: true
    llm_judge:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
    metrics:
      - faithfulness
      - answer_relevance
      - context_precision
      - context_recall
      - correctness
      - citation_accuracy
```

Run end-to-end evaluation:

```bash
rag evaluate rag.yaml
```

### Optimization

```yaml
optimization:
  primary_metric: faithfulness
  secondary_metric: latency_ms
  constraints:
    max_latency_ms: 500
  weights:
    faithfulness: 1.0
    latency_ms: -0.5
```

Run optimization on experiment results:

```bash
rag optimize rag.yaml
```

Export the recommended configuration:

```bash
rag export-config rag.yaml -o optimized.yaml
```

### Baseline Benchmark

```bash
rag benchmark ./data/docs ./data/queries.jsonl --version v1
```

The baseline v1 uses: recursive chunking (512/64), hash embeddings, dense retrieval (top_k=10), no reranker, mock generator.

## Configuration reference

| Section       | Field            | Default       | Description                        |
| ------------- | ---------------- | ------------- | ---------------------------------- |
| `project`     | `name`           | `rag-project` | Project name                       |
| `documents`   | `path`           | —             | Corpus directory or JSON file      |
| `chunking`    | `strategy`       | `recursive`   | Chunking strategy                  |
| `chunking`    | `chunk_size`     | `512`         | Target size per chunk              |
| `chunking`    | `overlap`        | `64`          | Overlap between adjacent chunks    |
| `embedding`   | `provider`       | `hash`        | Embedding provider name            |
| `embedding`   | `dimension`      | `128`         | Vector dimension (hash provider)   |
| `retrieval`   | `strategy`       | `dense`       | Retrieval strategy                 |
| `retrieval`   | `top_k`          | `5`           | Number of results returned         |
| `retrieval`   | `fusion`         | —             | Hybrid fusion settings             |
| `retrieval`   | `bm25`           | —             | BM25 k1/b/tokenizer settings       |
| `retrieval`   | `lambda_param`   | `0.5`         | MMR lambda (0=diversity, 1=relevance) |
| `reranker`    | `strategy`       | `none`        | Reranker strategy                  |
| `generation`  | `provider`       | `mock`        | Generation provider                |
| `generation`  | `model`          | `mock`        | Model name                         |
| `evaluation`  | `answer.enabled` | `true`        | Enable answer evaluation           |
| `optimization`| `primary_metric` | `mrr`         | Optimization primary metric        |
| `experiments` | `dataset`        | —             | Path to the JSONL query dataset    |
| `experiments` | `parameters`     | —             | Dot-path parameter sweeps          |
| `experiments` | `k`              | `10`          | Rank cutoff for metrics            |
| `experiments` | `primary_metric` | `mrr`         | Leaderboard sort metric            |
| `experiments` | `output_dir`     | `runs`         | Report output directory            |

`chunking.strategy` supports `recursive`, `fixed`, `sentence_window`, `parent_child`;
`retrieval.strategy` supports `dense`, `bm25`, `hybrid`, `mmr`;
`reranker.strategy` supports `cross_encoder`, `cohere`, `none`;
`generation.provider` supports `mock`, `openai`, `anthropic`, `ollama`.
Each strategy validates its own fields; unknown strategies or extra keys are rejected.

## Development

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest
```
