"""Configuration-driven experiment runner."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_sdk.chunking import build_chunker
from rag_sdk.config import ExperimentConfig, RagConfig
from rag_sdk.core import Document
from rag_sdk.embeddings import EmbeddingProvider, build_embedding_provider
from rag_sdk.evaluation import evaluate_retrieval
from rag_sdk.evaluation.relevance import judge_retrieval
from rag_sdk.experiments.dataset import QuerySample, load_queries
from rag_sdk.experiments.grid import expand_grid_with_detail
from rag_sdk.experiments.records import (
    EmbeddingInfo,
    ExperimentRecord,
    ExperimentResult,
    LatencyStats,
)
from rag_sdk.indexing import FaissVectorStore, InMemoryChunkStore, InMemoryDocumentStore
from rag_sdk.ingestion import IngestionResult, ingest_documents
from rag_sdk.reranking import RerankerProvider, build_reranker
from rag_sdk.retrieval import build_retrieval_pipeline


class ExperimentConfigWarning(UserWarning):
    """Emitted when a run's configuration makes its metrics misleading."""


class _CachingEmbeddingProvider(EmbeddingProvider):
    """Memoizes embeddings per text so repeated runs skip re-encoding.

    Caching is switched off while queries are timed (``caching = False``) so
    that search latency always includes query encoding and stays comparable
    across runs.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._cache: dict[str, np.ndarray] = {}
        self.caching = True

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self.caching:
            return self._provider.embed(texts)
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            vectors = self._provider.embed(missing)
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = vector
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._cache[text] for text in texts]).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._provider.dimension


def _cache_key(model: object) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True)  # type: ignore[attr-defined]


def _hash_file(path: Path) -> str:
    hash_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            hash_digest.update(block)
    return hash_digest.hexdigest()


class ExperimentRunner:
    """Runs every parameter combination and records comparable results.

    Pipeline construction is fully config-driven: chunking, embedding, and
    retrieval strategies are resolved from each variant's configuration.
    """

    def __init__(
        self,
        documents: list[Document],
        base_config: RagConfig,
        experiment: ExperimentConfig,
    ) -> None:
        if not documents:
            raise ValueError("cannot run an experiment without documents")
        self._documents = documents
        self._base_config = base_config
        self._experiment = experiment
        # Models are expensive to load; reuse them across variants that share
        # the same embedding / reranker configuration.
        self._embeddings: dict[str, _CachingEmbeddingProvider] = {}
        self._rerankers: dict[str, RerankerProvider] = {}

    def run(self) -> ExperimentResult:
        grid = expand_grid_with_detail(self._base_config, self._experiment.parameters)
        dataset_path = Path(self._experiment.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
        queries = load_queries(dataset_path)
        dataset_hash = _hash_file(dataset_path)
        records = [
            self._run_variant(
                variant.config,
                queries,
                dataset_path,
                dataset_hash,
                index,
                variant.skipped_parameters,
            )
            for index, variant in enumerate(grid)
        ]
        return ExperimentResult(
            dataset_path=str(dataset_path),
            dataset_hash=dataset_hash,
            primary_metric=self._experiment.primary_metric,
            k=self._experiment.k,
            relevance_level=self._experiment.relevance_level,
            records=records,
        )

    def _run_variant(
        self,
        config: RagConfig,
        queries: list[QuerySample],
        dataset_path: Path,
        dataset_hash: str,
        index: int,
        skipped_parameters: list[str],
    ) -> ExperimentRecord:
        chunk_store = InMemoryChunkStore()
        document_store = InMemoryDocumentStore()

        # Ingestion pipeline
        chunker = build_chunker(config.chunking)
        embedding = self._embedding_for(config)
        embedding.caching = True
        store = FaissVectorStore(dimension=embedding.dimension)

        # Ingest documents
        ingestion_result: IngestionResult = ingest_documents(
            self._documents,
            chunker,
            embedding,
            store,
            chunk_store=chunk_store,
            document_store=document_store,
            preprocessing_config=config.preprocessing,
        )

        # Build retrieval pipeline
        pipeline = build_retrieval_pipeline(
            config,
            embedding,
            store,
            chunk_store=chunk_store,
            document_store=document_store,
            chunks=ingestion_result.all_chunks,
            reranker=self._reranker_for(config),
        )

        run_warnings = self._check_config(config, indexed_chunks=len(store))
        for message in run_warnings:
            warnings.warn(f"run-{index}: {message}", ExperimentConfigWarning, stacklevel=2)

        # Build relevant_by_document from all chunks
        relevant_by_document: dict[str, list[str]] = {}
        for chunk in ingestion_result.all_chunks:
            relevant_by_document.setdefault(chunk.document_id, []).append(chunk.id)

        # Retrieve using pipeline
        search_top_k = max(config.retrieval.top_k, self._experiment.k)
        embedding.caching = False
        samples: list[tuple[list[str], set[str] | dict[str, int]]] = []
        latencies: list[float] = []
        for sample in queries:
            started = time.perf_counter()
            retrieved_results = pipeline.search(sample.query)
            latencies.append((time.perf_counter() - started) * 1000.0)
            samples.append(
                judge_retrieval(
                    retrieved_results[:search_top_k],
                    sample,
                    relevant_by_document,
                    self._experiment.relevance_level,
                )
            )

        metrics = evaluate_retrieval(samples, k=self._experiment.k)
        return ExperimentRecord(
            run_id=f"run-{index}",
            config=config.model_dump(mode="json"),
            dataset_path=str(dataset_path),
            dataset_hash=dataset_hash,
            timestamp=datetime.now(UTC),
            embedding=EmbeddingInfo(
                provider=config.embedding.provider,
                model=config.embedding.model,
                dimension=embedding.dimension,
            ),
            latency_ms=LatencyStats(
                mean_ms=statistics.fmean(latencies),
                median_ms=statistics.median(latencies),
            ),
            metrics=metrics,
            total_chunks=ingestion_result.total_chunks,
            skipped_parameters=skipped_parameters,
            warnings=run_warnings,
        )

    def _check_config(self, config: RagConfig, *, indexed_chunks: int) -> list[str]:
        """Flag configurations whose metrics would be silently misleading."""
        messages: list[str] = []
        k = self._experiment.k
        top_k = config.retrieval.top_k
        if top_k < k:
            messages.append(
                f"retrieval.top_k ({top_k}) < experiments.k ({k}): the pipeline "
                f"returns at most {top_k} results, so @{k} metrics are capped"
            )
        reranker = config.reranker
        if reranker is not None and reranker.strategy != "none":
            candidate_k = config.retrieval.candidate_k
            if candidate_k >= indexed_chunks:
                messages.append(
                    f"retrieval.candidate_k ({candidate_k}) >= indexed chunks "
                    f"({indexed_chunks}): the reranker sees the whole corpus, so "
                    f"retrieval strategies cannot be told apart"
                )
        return messages

    def _embedding_for(self, config: RagConfig) -> _CachingEmbeddingProvider:
        key = _cache_key(config.embedding)
        if key not in self._embeddings:
            self._embeddings[key] = _CachingEmbeddingProvider(
                build_embedding_provider(config.embedding)
            )
        return self._embeddings[key]

    def _reranker_for(self, config: RagConfig) -> RerankerProvider | None:
        if config.reranker is None:
            return None
        # ``top_k`` is read by the pipeline, not the reranker instance.
        key = json.dumps(
            config.reranker.model_dump(mode="json", exclude={"top_k"}), sort_keys=True
        )
        if key not in self._rerankers:
            reranker = build_reranker(config.reranker)
            if reranker is None:
                return None
            self._rerankers[key] = reranker
        return self._rerankers[key]


def run_experiment(documents: list[Document], base_config: RagConfig) -> ExperimentResult:
    """Run the experiment declared in ``base_config.experiments``."""
    if base_config.experiments is None:
        raise ValueError("configuration has no 'experiments' section")
    return ExperimentRunner(documents, base_config, base_config.experiments).run()