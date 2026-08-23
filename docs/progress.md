# Development Progress

Tracks completed phases and the tasks delivered in each. Phases follow
`docs/product-spec.md`; see `AGENTS.md` for the incremental strategy.

## Phase 1 — Project structure, config, chunking, FAISS, dense retrieval, evaluation, CLI (COMPLETE)

- Project scaffolding
  - `pyproject.toml` with hatchling packaging, dependencies, and `rag` console script
  - Ruff and pytest configuration wired into `pyproject.toml`
  - `src/` layout with subpackages: `core`, `config`, `chunking`, `embeddings`,
    `indexing`, `retrieval`, `evaluation`, `cli`
  - Removed the initial `main.py` stub

- Core domain types
  - `Document`, `Chunk`, `DocumentMetadata` with character offsets and serialization
  - Generic `Registry` for named component registration (plugin seed)

- Configuration system
  - Pydantic v2 models: `RagConfig`, `ProjectConfig`, `ChunkerConfig`,
    `RecursiveChunkerConfig`, `FixedTokenChunkerConfig`, `EmbeddingConfig`, `RetrievalConfig`
  - Discriminated union on `chunking.strategy` with per-strategy validation
  - Strict validation (`extra="forbid"`), overlap < chunk_size rule
  - YAML loader/dumper: `load_config`, `parse_config`, `dump_config`, `default_config`
  - Sample config at `configs/example.yaml`

- Chunking
  - `Chunker` ABC with `from_config` + `chunk(document)`
  - `RecursiveChunker` — hierarchical separator splitting with overlap
  - `FixedTokenChunker` — whitespace-token splitting with token overlap
  - Shared text primitives (`recursive_split`, `apply_overlap`, `whitespace_tokens`)
    producing char-offset-accurate pieces
  - `build_chunker` factory + `chunker_registry` / `register_chunker`

- Embeddings
  - `EmbeddingProvider` ABC (`embed`, `dimension`); real adapters deferred to later phases

- Indexing
  - `VectorStore` ABC (`add`, `search`, `len`)
  - `FaissVectorStore` (FAISS `IndexFlatIP`, normalized vectors = cosine similarity)

- Retrieval
  - `Retriever` ABC, `RetrievalResult`
  - `DenseRetriever` — embeds query, returns nearest chunks with metadata

- Evaluation
  - Retrieval metrics: `hit_at_k`, `recall_at_k`, `precision_at_k`, `reciprocal_rank`,
    `ndcg_at_k`, `average_precision`, `map`
  - `evaluate_retrieval` aggregate + `load_retrieval_results` (JSONL)

- CLI (thin)
  - `rag init` — write starter config
  - `rag validate` — validate a config and print a summary
  - `rag evaluate` — compute metrics from a JSONL results file

- Tests
  - Unit tests for core, config, chunking, FAISS store, dense retriever, metrics, CLI
  - End-to-end integration test: config -> chunk -> embed -> index -> retrieve -> evaluate
    (in-memory, no API keys)
  - `HashEmbeddingProvider` test fixture in `tests/conftest.py`

- Documentation
  - `docs/product-spec.md` (source of truth), `docs/index.md`, `docs/api.md`
  - `mkdocs.yml` (Material theme), updated `README.md`

- Verification
  - `ruff check .` clean
  - `pytest` — 66 tests passing on Python 3.14

## Phase 2 — Hybrid retrieval, BM25, experiment engine, reports (COMPLETE)

- Dependencies
  - Added `bm25s>=0.3.0` (pure NumPy BM25 backend, no scipy)

- Retrieval
  - `RetrievalConfig` is now a discriminated union on `strategy`:
    `DenseRetrievalConfig`, `BM25RetrievalConfig`, `HybridRetrievalConfig`
  - `BM25Params` (k1, b, tokenizer, stopwords) shared by BM25 and hybrid
  - `FusionConfig` (rrf | weighted, rrf_k, weights, candidate_k)
  - `BM25Retriever` — `bm25s`-backed lexical retrieval with chunk-id mapping
  - `HybridRetriever` — dense + BM25 with RRF or weighted fusion
  - Score fusion helpers: `rrf_fuse`, `weighted_fuse` (min-max normalized)
  - `build_retriever` factory + `retriever_registry`
  - `Retriever.add_chunks` promoted to the abstract interface

- Embeddings
  - `HashEmbeddingProvider` moved from tests into the SDK (offline, deterministic)
  - `build_embedding_provider` factory + `embedding_registry`; `EmbeddingConfig.dimension`

- Ingestion (minimal)
  - `load_documents` for `.txt`, `.md`, and `.json` corpora; `DocumentsConfig`
  - PDF/DOCX/HTML deferred to later phases

- Experiment engine
  - `ExperimentConfig` embedded in `RagConfig` with dot-path parameter sweeps
  - `expand_grid` — cartesian product over parameters; strategy selections
    apply variant defaults; inapplicable overrides are skipped per combination
    with an `ExperimentParameterWarning` and recorded in the run's
    `skipped_parameters` metadata (visible in CSV/JSON/HTML reports)
  - `ExperimentRunner` — config-driven chunk → embed → index → retrieve →
    evaluate per variant; records config, dataset hash, metrics, latency,
    timestamp, embedding info
  - `ExperimentRecord` / `ExperimentResult` (Pydantic), JSONL dataset loader
  - Reports: CSV, JSON, leaderboard CSV, self-contained interactive HTML
    (vanilla JS sorting) + best-config recommendation by `primary_metric` (MRR default)

- CLI (thin)
  - `rag experiment CONFIG [--output DIR]` — runs the sweep and writes reports

- Config
  - `documents:` and `experiments:` sections; `dump_config` excludes nulls
  - Example sweep config at `configs/experiment.yaml`

- Tests
  - Unit tests: BM25, fusion, hybrid, hash embeddings, ingestion, grid,
    dataset, runner, reports, experiment config models, `rag experiment` CLI
  - Integration: BM25 and hybrid retrieval pipelines; full config-driven
    experiment producing all four report files
  - `pytest` — 124 tests passing on Python 3.14

## Phase 3 — Sentence window, parent-child, auto-merging, reranking (PENDING)

## Phase 4 — Answer evaluation, observability, production runtime (PENDING)
