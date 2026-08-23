# API Reference

Anything exposed from the top-level package and its submodules is public API.

## `rag_sdk`

- `Document(id, text, metadata)` — a source document.
- `Chunk(id, text, document_id, index, start_char, end_char, metadata)` — a
  contiguous slice of a document with character offsets.
- `DocumentMetadata(source, title, page, headings)` — preserved metadata.
- `Registry[T]` — generic named component registry with `.register`, `.get`,
  `.names`, and a `.decorator` for registering custom implementations.
- `Citation` — citation linking answer to source (document_id, chunk_id, page, section, source_uri, text_span, score).
- `CitedAnswer` — answer text with list of citations.

## `rag_sdk.config`

- `load_config(path) -> RagConfig` — load and validate a YAML config.
- `parse_config(text) -> RagConfig` — validate a YAML string.
- `dump_config(config) -> str` — serialize a config to YAML.
- `default_config() -> RagConfig` — the `rag init` skeleton.
- `baseline_config(version) -> RagConfig` — versioned baseline for benchmarking.
- `RagConfig`, `ProjectConfig`, `ChunkerConfig`, `RecursiveChunkerConfig`,
  `FixedTokenChunkerConfig`, `SentenceWindowChunkerConfig`, `ParentChildChunkerConfig`,
  `EmbeddingConfig`, `DocumentsConfig`, `ExperimentConfig`, `RetrievalConfig` (a discriminated union of
  `DenseRetrievalConfig`, `BM25RetrievalConfig`, `HybridRetrievalConfig`, `MMRRetrievalConfig`),
  `BM25Params`, `FusionConfig`, `MMRRetrievalConfig`,
  `GenerationConfig` (discriminated union: `MockGenerationConfig`, `OpenAIGenerationConfig`,
  `AnthropicGenerationConfig`, `OllamaGenerationConfig`),
  `JudgeConfig`, `AnswerEvaluationConfig`, `EvaluationConfig`, `OptimizationConfig`,
  `TelemetryConfig`, `CaptureConfig`, `CacheConfig`, `TokenizerConfig`.

## `rag_sdk.chunking`

- `Chunker` — abstract base; implementations must provide `from_config` and
  `chunk(document)`.
- `RecursiveChunker` — hierarchical separator splitting with overlap.
- `FixedTokenChunker` — whitespace-token splitting with token overlap.
- `SentenceWindowChunker` — overlapping sentence windows as chunks.
- `ParentChildChunker` — hierarchical parent/child chunks.
- `build_chunker(config) -> Chunker` — resolve a chunker by strategy name.
- `chunker_registry` / `register_chunker` — register custom chunkers.

## `rag_sdk.embeddings`

- `EmbeddingProvider` — abstract base; `embed(texts) -> np.ndarray` and
  `dimension`.
- `HashEmbeddingProvider` — deterministic, offline provider for development.
- `build_embedding_provider(config)` — resolve a provider by name.
- `embedding_registry` / `register_embedding` — register custom providers.

## `rag_sdk.ingestion`

- `load_documents(path)` — load `.txt`, `.md`, or `.json` corpora into
  `Document` objects.
- `IngestionError` — raised for missing or malformed corpora.
- `ingest_documents(documents, chunker, embedding, store, chunk_store, document_store)` —
  full ingestion pipeline with sentence boundary computation and store persistence.

## `rag_sdk.indexing`

- `VectorStore` — abstract base; `add(ids, vectors)`, `search(vector, k)`, `len`,
  `get_embedding(chunk_id)`.
- `FaissVectorStore` — in-memory FAISS store using inner-product similarity.
  Pass normalized vectors so scores equal cosine similarity.
- `DocumentStore` / `ChunkStore` — abstractions with `SQLiteStore` and `InMemoryStore` implementations.

## `rag_sdk.retrieval`

- `Retriever` — abstract base; `add_chunks(chunks)` and `search(query, top_k)`.
- `DenseRetriever(embedding_provider, store)` — embed the query and return the
  nearest chunks.
- `BM25Retriever(params=None)` — `bm25s`-backed lexical retrieval.
- `HybridRetriever(dense, lexical, config=None)` — fused dense + BM25.
- `MMRRetriever(config, embedding_provider, store)` — Maximal Marginal Relevance retrieval.
- `RetrievalResult(query, chunk, score)` with lineage fields.
- `rrf_fuse`, `weighted_fuse` — score fusion helpers.
- `ParentChildExpander`, `SentenceWindowExpander`, `AutoMerger` — enrichment components.
- `RetrievalPipeline` — composes retriever, reranker, and enrichers.
- `build_retriever(config, embedding_provider, store)` — resolve a retriever by
  strategy name; `retriever_registry` / `register_retriever`.
- `build_retrieval_pipeline(...)` — full pipeline factory.

## `rag_sdk.reranking`

- `RerankerProvider` — abstract base; `rerank(query, candidates, top_k)`.
- `CrossEncoderReranker` — sentence-transformers cross-encoder (e.g., BAAI/bge-reranker-base).
- `CohereReranker` — Cohere Rerank v4 API.
- `NoOpReranker` — baseline for experiments.
- `build_reranker(config)` — resolve a reranker by strategy name.

## `rag_sdk.generation`

- `Generator` — abstract base; `generate(prompt, context, **params) -> GenerationResponse`.
- `MockGenerator` — deterministic canned responses for testing.
- `OpenAIGenerator` — OpenAI chat completions API.
- `AnthropicGenerator` — Anthropic Messages API.
- `OllamaGenerator` — local Ollama server.
- `GenerationResponse` — text, `cited_answer`, token usage, model metadata.
- `CitedAnswer` / `Citation` — answer with inline citations.
- `build_generator(config)` — resolve a generator by provider name; `generator_registry` / `register_generator`.

## `rag_sdk.tokenizer`

- `Tokenizer` — protocol: `count(text)`, `encode(text)`, `decode(tokens)`.
- `WhitespaceTokenizer` — built-in, no dependencies.
- `Cl100kBaseTokenizer` — OpenAI's cl100k_base via tiktoken (optional).
- `TokenizerConfig` — discriminated union: whitespace, cl100k_base, custom.
- `build_tokenizer(config)` — resolve a tokenizer by type.

## `rag_sdk.context`

- `ContextBuilder` — protocol: `build(retrieved_chunks, query, config) -> Context`.
- `DefaultContextBuilder` — deduplication, token budgeting, citation formatting.
- `Context` — text, token_count, source_chunks, tokenizer_type.
- `ContextConfig` — max_tokens, include_metadata, citation_format, deduplicate, tokenizer.
- `build_context_builder(config)` — resolve a context builder.

## `rag_sdk.evaluation`

- `hit_at_k`, `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k`,
  `average_precision`, `map` — per-query and aggregate retrieval metrics.
- `evaluate_retrieval(results, k)` — aggregate all metrics across queries.
- `load_retrieval_results(path)` — load JSONL results for evaluation.
- `load_evaluation_samples(path)` — load JSONL with reference answers.

### Answer Evaluation

- `Evaluator` — protocol: `evaluate(sample, result) -> EvaluationResult`.
- `EvaluationSample` — query_id, query, relevant_documents, relevant_chunks, reference_answer.
- `RAGResult` — query, query_id, retrieved_chunks, generation, retrieval_metrics.
- `EvaluationResult` — metric_name, score, reason, evaluator_metadata, errors.

Reference-based evaluators:
- `FaithfulnessEvaluator` — token overlap answer vs reference.
- `AnswerRelevanceEvaluator` — query token overlap in answer.
- `ContextPrecisionEvaluator` — relevant chunks in top-K.
- `ContextRecallEvaluator` — fraction of relevant chunks retrieved.
- `CorrectnessEvaluator` — fuzzy match vs reference answer.
- `CitationAccuracyEvaluator` — citations point to relevant chunks.

LLM-as-judge evaluators (independent judge config):
- `LLMFaithfulnessEvaluator`, `LLMAnswerRelevanceEvaluator`, `LLMCorrectnessEvaluator`.

- `build_evaluators(config, judge_generator)` — build evaluator list from config.
- `EvaluationPipeline(config, documents).evaluate(dataset_path)` — full pipeline.
- `evaluate_rag(config, documents, dataset_path)` — convenience function.

## `rag_sdk.optimization`

- `Optimizer` — abstract base; `optimize(results, config) -> OptimizationResult`.
- `ParetoOptimizer` — multi-objective Pareto optimization with constraints.
- `OptimizationResult` — recommended_config, reasoning, pareto_frontier, baseline_comparison, all_configs.
- `ParetoPoint` — config, metrics, dominated flag.
- `build_optimizer(type) -> Optimizer` — factory.

## `rag_sdk.telemetry`

- `TelemetryConfig` — enabled, capture (prompts, responses, retrieved_content, document_content).
- `TelemetryContext` — thread-local context for experiment run; `emit(name, attributes)`.
- `TelemetryEvent` — name, timestamp, attributes.
- `ConsoleExporter` — exports events as JSONL to stdout.
- `get_telemetry_context()` / `set_telemetry_context(context)` — context variable access.

## `rag_sdk.cache`

- `Cache` — protocol: `get(key)`, `set(key, value, ttl)`, `delete(key)`, `clear()`.
- `InMemoryCache` — dict-based with TTL support.
- `CacheConfig` — enabled, ttl_seconds.

## `rag_sdk.experiments`

- `ExperimentConfig` — dataset, dot-path parameter sweeps, rank cutoff,
  primary metric, output directory.
- `expand_grid(base, parameters)` / `apply_override(config, path, value)` —
  build the parameter sweep cartesian product.
- `expand_grid_with_detail(base, parameters) -> list[GridVariant]` — as above
  but each variant reports the parameters skipped for its strategy.
- `ExperimentParameterWarning` — emitted when a sweep parameter does not apply
  to a strategy (for example `retrieval.fusion.method` on a `dense` variant).
- `ExperimentRunner(documents, base_config, experiment).run() ->
  ExperimentResult` — run every combination.
- `run_experiment(documents, base_config)` — run the sweep declared in config.
- `ExperimentRecord` / `ExperimentResult` — per-run and aggregate results;
  each record carries `skipped_parameters`; `result.leaderboard()` sorts by
  the primary metric.
- `load_queries(path)` — load a JSONL dataset
  (`{"query", "relevant_documents": [doc_id, ...]}`).
- `write_csv`, `write_json`, `write_leaderboard`, `write_html`,
  `write_reports(result, output_dir)` — report writers.

## `rag_sdk.cli`

- `app` — the `rag` Typer application with `init`, `validate`, `evaluate`,
  `benchmark`, `experiment`, `optimize`, `export-config` commands.
  The CLI stays thin; logic lives in the SDK modules.
