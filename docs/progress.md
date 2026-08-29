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

## Phase 3 — Sentence window, parent-child, auto-merging, reranking (COMPLETE)

- Chunking
  - `SentenceWindowChunker` — overlapping sentence windows as chunks
  - `ParentChildChunker` — hierarchical parent/child chunks (side-effect free)
    - Children indexed in VectorStore, parents persisted to ChunkStore
    - Explicit `parent_id` / `child_ids` metadata on chunks
  - Rule-based sentence splitter (no NLTK dependency) with abbreviation handling
  - Sentence boundary computation stored in chunk metadata for retrieval-time expansion

- Storage
  - `DocumentStore` abstraction (SQLite + InMemory) for full document retrieval
  - `ChunkStore` abstraction (SQLite + InMemory) for parent chunk persistence
  - Per-experiment-run store paths (`run-{index}/chunkstore.db`, `run-{index}/docstore.db`)

- Ingestion pipeline (`ingest_documents`)
  - Stores full documents in DocumentStore
  - Computes and stores sentence boundaries for all chunks
  - ParentChildChunker: embeds/indexes children only; persists parents to ChunkStore
  - Standard chunkers: embeds/indexes all chunks

- VectorStore extension
  - `get_embedding(chunk_id)` for retrieving stored embeddings (used by auto-merging)

- Reranking
  - `RerankerProvider` ABC with `rerank(query, candidates, top_k)`
  - `CrossEncoderReranker` — sentence-transformers cross-encoder (e.g., BAAI/bge-reranker-base)
  - `CohereReranker` — Cohere Rerank v4 API (model: rerank-v4.0-fast)
  - `NoOpReranker` — baseline for experiment comparisons (`strategy: none`)
  - Single canonical `reranker:` config at `RagConfig` level (not under RetrievalConfig)

- Retrieval pipeline (Retriever → Reranker → Enrichment)
  - `RetrievalPipeline` composes retriever, reranker, and enrichers
  - Retriever returns `candidate_k` results; reranker returns `reranker.top_k`
  - No reranker: truncates to `retrieval.top_k` BEFORE enrichment
  - Lineage preserved: `source_chunk_id`, `source_chunk_score`, `source_chunk_rank`
  - Reranker lineage: `rerank_score`, `rerank_rank`

- Context enrichment (after reranking)
  - `ParentChildExpander` — maps source chunk → parent via ChunkStore
    - Deduplicates parent chunks; preserves all child IDs in `child_ids`
    - Aggregates scores/ranks from contributing children
  - `SentenceWindowExpander` — expands using stored boundaries + DocumentStore
    - Resolves original source chunk via `source_chunk_id`
    - Uses sentence boundaries from ingestion-time computation
  - `AutoMerger` — merges adjacent chunks from same document
    - Uses stored embeddings from VectorStore via `source_chunk_id`
    - Token-aware (`max_tokens`) with configurable tokenizer (whitespace/cl100k_base)
    - Preserves `merged_source_ids` lineage

- Configuration
  - Explicit enrichment configs under `retrieval:` (not inferred from chunker)
  - `candidate_k` (retriever pool), `reranker.top_k`, `retrieval.top_k` separate
  - `RerankerConfig` discriminated union: cross_encoder | cohere | none

- Experiments
  - Sweeps over `reranker.strategy: [none, cross_encoder, cohere]`
  - Sweeps over enrichment flags: sentence_window, parent_child, auto_merging
  - Per-run stores, full lineage in reports

- Tests
  - All 128 tests passing (unit + integration)
  - Cartesian sweeps, inapplicable overrides, parent deduplication
  - Lineage preservation, score preservation, pipeline ordering
  - Enrichment combinations: parent_child+auto_merging, sentence_window+parent_child, all three

- Example config: `configs/phase3.yaml`

## Phase 4 — Answer evaluation, observability, optimization, caching (COMPLETE)

- Generation engine (for evaluation only)
  - `Generator` ABC with `generate(prompt, context)` → `GenerationResponse`
  - Providers: `MockGenerator`, `OpenAIGenerator`, `AnthropicGenerator`, `OllamaGenerator`
  - `build_generator` factory + `generator_registry`
  - Optional deps: `openai`, `anthropic`, `httpx` via `[generation]` extra

- Citation support
  - `Citation` — document_id, chunk_id, page, section, source_uri, text_span, score
  - `CitedAnswer` — answer text + list of citations
  - Lineage preserved from retrieval → context → generation

- Context construction (separate stage per spec)
  - `ContextBuilder` ABC with `build(retrieved_chunks, query, config)` → `Context`
  - `DefaultContextBuilder` — deduplication, token budgeting, metadata formatting
  - Config: `max_tokens`, `include_metadata`, `citation_format`, `tokenizer`
  - Tokenizer abstraction: `WhitespaceTokenizer`, `Cl100kBaseTokenizer` (optional tiktoken)

- Answer evaluation (retrieval + generation + answer eval)
  - `Evaluator` protocol — `evaluate(sample, result)` → `EvaluationResult`
  - Reference-based evaluators: `FaithfulnessEvaluator`, `AnswerRelevanceEvaluator`,
    `ContextPrecisionEvaluator`, `ContextRecallEvaluator`, `CorrectnessEvaluator`,
    `CitationAccuracyEvaluator`
  - LLM-as-judge evaluators (independent judge config): `LLMFaithfulnessEvaluator`,
    `LLMAnswerRelevanceEvaluator`, `LLMCorrectnessEvaluator`
  - `EvaluationPipeline` — composes retriever → reranker → context → generator → evaluators
  - JSONL dataset with `query_id`, `query`, `relevant_documents`, `relevant_chunks`, `reference_answer`

- Optimization engine
  - `ParetoOptimizer` — identifies Pareto-optimal configs across objectives
  - Objectives: retrieval quality, answer quality, latency, token usage, estimated cost
  - Constraints: `max_latency_ms`, `max_cost`, `min_recall`
  - Weighted scoring with primary/secondary metrics
  - `OptimizationResult` — recommended_config, reasoning, pareto_frontier, baseline_comparison

- Baseline configuration & `rag benchmark`
  - Versioned baseline (v1): recursive chunking + hash embeddings + dense retrieval + mock generator
  - Reproducible for consistent comparison across experiments

- Minimal telemetry (optional, disabled by default)
  - Structured event logging: `stage_latency`, `token_usage`, `experiment_start`, `variant_start`, `error`
  - Console/JSONL exporter; privacy-first `CaptureConfig` (prompts, responses, content off by default)
  - OTLP/OpenTelemetry as optional `[observability]` extra

- Cache abstraction
  - `Cache` protocol: `get`, `set`, `delete`, `clear`
  - `InMemoryCache` with TTL support
  - Integration points: embedding, retrieval, generation caches (config-gated)

- CLI commands
  - `rag evaluate CONFIG` — end-to-end evaluation (retrieval + generation + answer eval)
  - `rag benchmark DOCUMENTS DATASET` — run versioned baseline, save results
  - `rag experiment CONFIG` — parameter sweeps with reports
  - `rag optimize CONFIG` — Pareto optimization on experiment results, print recommendation
  - `rag export-config CONFIG` — export optimized config as YAML/JSON

- Tests
  - 176 tests passing (unit + integration)
  - New unit tests: MMR, tokenizer, generation, context, answer eval, optimization, telemetry, cache
  - Integration: full evaluation pipeline, experiment + optimization flow, all CLI commands
  - No external API keys required for core tests (mock providers)

- Documentation
  - Updated `docs/progress.md`, `README.md`, `docs/api.md`, `docs/index.md`
  - Example configs: `configs/phase4.yaml` (full pipeline with generation + eval + optimization)

## Phase 5 — Document ingestion, preprocessing, advanced chunking, advanced retrieval (COMPLETE)

- Document Ingestion
  - `DocumentLoader` ABC with pluggable registry (`loader_registry`, `build_loader`)
  - `PyPDFLoader` — PDF extraction using `pypdf` (page-level metadata)
  - `DocxLoader` — DOCX extraction using `python-docx` (heading hierarchy)
  - `HTMLLoader` — HTML extraction using `BeautifulSoup` + `readability-lxml` (main content, title, headings)
  - `TextLoader`, `JSONLoader` — existing loaders now in registry
  - Config: `DocumentsConfig` with `loader`, `recursive`, `glob_pattern`

- Preprocessing Pipeline
  - `Preprocessor` ABC with pluggable registry (`preprocessor_registry`)
  - `WhitespaceNormalizer` — collapse whitespace, normalize newlines
  - `TextCleanup` — remove control chars, fix encoding, unicode normalization
  - `HeaderFooterRemover` — detect repeated headers/footers across docs
  - `DuplicateDetector` — exact hash deduplication (semantic optional)
  - `MetadataExtractor` — extract emails, URLs, dates, titles from text
  - Fixed pipeline order: normalize → cleanup → headers → duplicates → metadata
  - Config: `PreprocessingConfig` under `RagConfig.preprocessing`

- Semantic Chunking
  - `SemanticChunker` — embedding-based similarity threshold chunking
  - Algorithm: sentence split → embed → cosine similarity → group by threshold
  - Config: `SemanticChunkerConfig` (similarity_threshold, min/max_chunk_size)
  - Uses global `RagConfig.embedding` provider
  - Respects size constraints with fallback splitting

- Structure-Aware Chunking
  - `StructureAwareChunker` — respects heading hierarchy from ingestion
  - Preserves section hierarchy in chunk metadata (`section_hierarchy`)
  - Config: `StructureAwareChunkerConfig` (include_heading_context, max_heading_depth)

- Metadata Filtering
  - First-class `filters` dict in `RetrievalConfigBase`
  - Native FAISS metadata filtering with post-filter fallback
  - Supported in Dense, BM25, Hybrid, MMR retrievers

- Multi-Query Retrieval
  - `MultiQueryRetriever` wrapper generating N queries from one
  - LLM-based query generation using `RagConfig.generation`
  - RRF or weighted fusion of results
  - Config: `MultiQueryConfig` under `RetrievalConfigBase`

- Query Rewriting
  - `QueryRewriter` ABC with pluggable registry
  - `LLMQueryRewriter` — LLM-based query expansion/rewriting
  - `TemplateQueryRewriter` — template-based transformation
  - `HyDEQueryRewriter` — Hypothetical Document Embeddings
  - Integrated in retrieval pipeline before embedding
  - Config: `QueryRewriterConfig` under `RetrievalConfigBase`

- Graded Relevance & Dataset Versioning
  - `QuerySample.relevance_grades` for chunk-level graded relevance (0-3)
  - `EvaluationDataset` with `version`, `metadata`, `samples`
  - `load_dataset()` for full dataset with metadata
  - `ndcg_at_k` supports graded relevance, added `map_graded` metric

- External Tests
  - `tests/external/` with real PDF/DOCX/HTML fixtures
  - Download script: `tests/external/download_fixtures.py`
  - Run with: `uv run pytest tests/external -m external`
  - Semantic chunking quality tests with real embeddings

- Documentation
  - Example configs: `pdf_ingestion.yaml`, `docx_ingestion.yaml`, `html_ingestion.yaml`,
    `semantic_chunking.yaml`, `structure_aware.yaml`, `metadata_filtering.yaml`,
    `multi_query.yaml`, `query_rewriting.yaml`

- Tests
  - All 200+ tests passing (unit + integration + external)
  - External tests with real documents for extraction quality verification

- Verification
  - `ruff check .` clean
  - `pytest` — 200+ tests passing on Python 3.14
