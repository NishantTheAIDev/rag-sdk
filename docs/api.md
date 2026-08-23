# API Reference

Anything exposed from the top-level package and its submodules is public API.

## `rag_sdk`

- `Document(id, text, metadata)` — a source document.
- `Chunk(id, text, document_id, index, start_char, end_char, metadata)` — a
  contiguous slice of a document with character offsets.
- `DocumentMetadata(source, title, page, headings)` — preserved metadata.
- `Registry[T]` — generic named component registry with `.register`, `.get`,
  `.names`, and a `.decorator` for registering custom implementations.

## `rag_sdk.config`

- `load_config(path) -> RagConfig` — load and validate a YAML config.
- `parse_config(text) -> RagConfig` — validate a YAML string.
- `dump_config(config) -> str` — serialize a config to YAML.
- `default_config() -> RagConfig` — the `rag init` skeleton.
- `RagConfig`, `ProjectConfig`, `ChunkerConfig`, `RecursiveChunkerConfig`,
  `FixedTokenChunkerConfig`, `EmbeddingConfig`, `DocumentsConfig`,
  `ExperimentConfig`, `RetrievalConfig` (a discriminated union of
  `DenseRetrievalConfig`, `BM25RetrievalConfig`, `HybridRetrievalConfig`),
  `BM25Params`, `FusionConfig`.

## `rag_sdk.chunking`

- `Chunker` — abstract base; implementations must provide `from_config` and
  `chunk(document)`.
- `RecursiveChunker` — hierarchical separator splitting with overlap.
- `FixedTokenChunker` — whitespace-token splitting with token overlap.
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

## `rag_sdk.indexing`

- `VectorStore` — abstract base; `add(ids, vectors)`, `search(vector, k)`, `len`.
- `FaissVectorStore` — in-memory FAISS store using inner-product similarity.
  Pass normalized vectors so scores equal cosine similarity.

## `rag_sdk.retrieval`

- `Retriever` — abstract base; `add_chunks(chunks)` and `search(query, top_k)`.
- `DenseRetriever(embedding_provider, store)` — embed the query and return the
  nearest chunks.
- `BM25Retriever(params=None)` — `bm25s`-backed lexical retrieval.
- `HybridRetriever(dense, lexical, config=None)` — fused dense + BM25.
- `RetrievalResult(query, chunk, score)`.
- `rrf_fuse`, `weighted_fuse` — score fusion helpers.
- `build_retriever(config, embedding_provider, store)` — resolve a retriever by
  strategy name; `retriever_registry` / `register_retriever`.

## `rag_sdk.evaluation`

- `hit_at_k`, `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k`,
  `average_precision`, `map` — per-query and aggregate retrieval metrics.
- `evaluate_retrieval(results, k)` — aggregate all metrics across queries.
- `load_retrieval_results(path)` — load JSONL results for evaluation.

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

- `app` — the `rag` Typer application with `init`, `validate`, `evaluate`, and
  `experiment` commands. The CLI stays thin; logic lives in the SDK modules.
