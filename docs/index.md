# RAG SDK

A Python SDK for building, experimenting with, evaluating, and optimizing
production-grade RAG pipelines through configuration instead of boilerplate.

The primary goal is to answer **"which RAG configuration works best for my
data?"** by making chunking, embedding, retrieval, reranking, generation,
evaluation, and optimization configurable and measurable.

## Features

- **Document Ingestion**: PDF (pypdf), DOCX (python-docx), HTML (BeautifulSoup + readability-lxml), text, JSON, Markdown
- **Preprocessing Pipeline**: whitespace normalization, text cleanup, header/footer removal, deduplication, metadata extraction
- **Chunking**: recursive, fixed-token, sentence window, parent-child, **semantic**, **structure-aware**
- **Embeddings**: Hash (deterministic), Sentence Transformers, pluggable interface
- **Vector Store**: FAISS (IndexFlatIP, normalized = cosine similarity)
- **Retrieval**: dense, BM25, hybrid (RRF/weighted fusion), MMR
- **Metadata Filtering**: first-class filters in retrieval config
- **Multi-Query Retrieval**: LLM-generated query expansion with RRF/weighted fusion
- **Query Rewriting**: LLM, template, HyDE (Hypothetical Document Embeddings)
- **Reranking**: cross-encoder, Cohere v4, or none (baseline)
- **Context Enrichment**: parent-child expansion, sentence window, auto-merging
- **Generation** (for evaluation): Mock, OpenAI, Anthropic, Ollama
- **Citation Support**: preserve source lineage through to generated answers
- **Context Construction**: token-budgeted, deduplicated, metadata-aware
- **Answer Evaluation**: reference-based + LLM-as-judge (faithfulness, relevance, precision, recall, correctness, citation accuracy)
- **Graded Relevance**: nDCG and MAP with 0-3 relevance grades, dataset versioning
- **Optimization**: Pareto frontier, constraints, weighted scoring, baseline comparison
- **Metrics**: Hit@K, Recall@K, Precision@K, MRR, nDCG, MAP (binary + graded)
- **Experiments**: config-driven parameter sweeps with CSV/JSON/HTML reports
- **CLI**: `init`, `validate`, `evaluate`, `evaluate-pipeline`, `benchmark`, `experiment`, `optimize`, `export-config`

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
  relevance_level: document  # document | chunk
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

How sweeps are applied:

- Selecting a strategy (`chunking.strategy`, `retrieval.strategy`,
  `reranker.strategy`) swaps in that strategy's defaults but keeps the fields
  shared by every variant, such as `top_k`, `candidate_k`, `filters`,
  `chunk_size` and `overlap`. Sweeping `reranker.strategy` creates the
  reranker section if the base config has none.
- Each combination is validated once after all overrides are applied, so
  `chunking.chunk_size: [64]` together with `chunking.overlap: [16]` works in
  any declaration order. An invalid combination fails with the parameter
  values that caused it.
- Embedding models and rerankers are loaded once and shared by every
  combination with the same settings, and identical texts are embedded once.
  Set `HF_HUB_OFFLINE=1` once Hugging Face models are cached to skip the hub
  network checks.

Relevance and metrics:

- `relevance_level: document` (default) collapses retrieved chunks to their
  documents before scoring against `relevant_documents`, so metrics are
  comparable across chunk sizes.
- `relevance_level: chunk` scores chunk IDs against `relevant_chunks` and
  `relevance_grades` when a query provides them; otherwise every chunk of a
  relevant document counts as relevant, which favours fewer, larger chunks.
  Chunk IDs depend on the chunker, so chunk-level labels only make sense when
  chunking is not swept.

The runner warns (and records the message in the run's `warnings`) when a
configuration makes its metrics misleading:

- `retrieval.top_k` < `experiments.k`: the pipeline returns at most `top_k`
  results, so @k metrics are capped.
- A reranker is enabled and `retrieval.candidate_k` covers every indexed
  chunk: the reranker sees the whole corpus, so retrieval strategies cannot
  be told apart.

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

Run end-to-end evaluation (index `documents.path`, retrieve, generate, score):

```bash
rag evaluate-pipeline rag.yaml --dataset queries.jsonl -o runs/eval
```

The dataset defaults to `evaluation.dataset`, then `experiments.dataset`, and
the rank cutoff to `evaluation.k`, then `retrieval.top_k`. Retrieval metrics
(Hit/Precision/Recall@k, MRR, nDCG, MAP, at `evaluation.relevance_level`) and
answer metrics are reported separately, along with retrieval and total latency.
With `-o`, `evaluation.json` records the config, dataset path and hash, metrics,
latency, timestamp, model information and a per-query breakdown. Generation and
LLM judges may call paid APIs depending on their providers. (`rag evaluate`
only scores an existing retrieval-results JSONL file.)

```yaml
evaluation:
  dataset: ./queries.jsonl   # optional
  k: 5                       # optional, defaults to retrieval.top_k
  relevance_level: document  # document | chunk
```

The same run from Python:

```python
from rag_sdk.config import load_config
from rag_sdk.evaluation import evaluate_rag, write_evaluation_report
from rag_sdk.ingestion import load_documents

config = load_config("rag.yaml")
documents = load_documents(config.documents.path, config.documents.loader)
result = evaluate_rag(config, documents, "queries.jsonl")
print(result["retrieval_metrics"], result["answer_metrics"])
write_evaluation_report(result, "runs/eval/evaluation.json")
```

### Optimization

```yaml
optimization:
  primary_metric: faithfulness
  secondary_metric: latency_ms
  constraints:
    latency_ms: 500        # keys are metric names; latency is an upper bound
  weights:
    faithfulness: 1.0
    latency_ms: -0.5
```

Run optimization on experiment results:

```bash
rag optimize rag.yaml
```

`rag optimize` reads `results.json` from `experiments.output_dir` (or `-o DIR`),
keeps the Pareto frontier (quality metrics maximized, mean `latency_ms`
minimized), applies `constraints` and `weights`, and writes the full
recommended pipeline to `<output_dir>/optimized-rag.yaml`. Any retrieval
metric (`hit_at_k`, `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`, `map`)
or `latency_ms` can be the primary or secondary metric. Answer metrics such as
`faithfulness` are only available from end-to-end evaluation, not from
`rag experiment` results.

Convert the recommended configuration to JSON:

```bash
rag export-config runs/optimized-rag.yaml -f json -o optimized.json
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
| `experiments` | `relevance_level`| `document`    | Score documents or chunks          |
| `evaluation`  | `dataset`        | —             | Query set for `evaluate-pipeline`  |
| `evaluation`  | `k`              | `top_k`       | Rank cutoff for pipeline metrics   |
| `evaluation`  | `relevance_level`| `document`    | Score documents or chunks          |
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
