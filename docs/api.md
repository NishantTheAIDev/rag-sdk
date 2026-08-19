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
  `FixedTokenChunkerConfig`, `EmbeddingConfig`, `RetrievalConfig`.

## `rag_sdk.chunking`

- `Chunker` — abstract base; implementations must provide `from_config` and
  `chunk(document)`.
- `RecursiveChunker` — hierarchical separator splitting with overlap.
- `FixedTokenChunker` — whitespace-token splitting with token overlap.
- `build_chunker(config) -> Chunker` — resolve a chunker by strategy name.
- `chunker_registry` / `register_chunker` — register custom chunkers.

## `rag_sdk.embeddings`

- `EmbeddingProvider` — abstract base; `embed(texts) -> np.ndarray` and
  `dimension`. Provider-specific adapters will live behind this interface.

## `rag_sdk.indexing`

- `VectorStore` — abstract base; `add(ids, vectors)`, `search(vector, k)`, `len`.
- `FaissVectorStore` — in-memory FAISS store using inner-product similarity.
  Pass normalized vectors so scores equal cosine similarity.

## `rag_sdk.retrieval`

- `Retriever` — abstract base; `search(query, top_k)`.
- `DenseRetriever(embedding_provider, store)` — embed the query and return the
  nearest chunks. Register chunks with `add_chunks(chunks)`.
- `RetrievalResult(query, chunk, score)`.

## `rag_sdk.evaluation`

- `hit_at_k`, `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k`,
  `average_precision`, `map` — per-query and aggregate retrieval metrics.
- `evaluate_retrieval(results, k)` — aggregate all metrics across queries.
- `load_retrieval_results(path)` — load JSONL results for evaluation.

## `rag_sdk.cli`

- `app` — the `rag` Typer application with `init`, `validate`, and `evaluate`
  commands. The CLI stays thin; logic lives in the SDK modules.
