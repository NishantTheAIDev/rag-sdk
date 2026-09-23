"""Configuration-driven experiment runner."""

from __future__ import annotations

import hashlib
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from rag_sdk.chunking import build_chunker
from rag_sdk.config import ExperimentConfig, RagConfig
from rag_sdk.core import Document
from rag_sdk.embeddings import build_embedding_provider
from rag_sdk.evaluation import evaluate_retrieval
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
from rag_sdk.retrieval import build_retrieval_pipeline


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
        # Create per-run stores
        run_dir = Path(self._experiment.output_dir) / f"run-{index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        chunk_store = InMemoryChunkStore()
        document_store = InMemoryDocumentStore()

        # Ingestion pipeline
        chunker = build_chunker(config.chunking)
        embedding = build_embedding_provider(config.embedding)
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
        )

        # Build relevant_by_document from all chunks
        relevant_by_document: dict[str, list[str]] = {}
        for chunk in ingestion_result.all_chunks:
            relevant_by_document.setdefault(chunk.document_id, []).append(chunk.id)

        # Retrieve using pipeline
        search_top_k = max(config.retrieval.top_k, self._experiment.k)
        samples: list[tuple[list[str], set[str]]] = []
        latencies: list[float] = []
        for sample in queries:
            started = time.perf_counter()
            retrieved_results = pipeline.search(sample.query)
            retrieved_ids = [r.chunk.id for r in retrieved_results[:search_top_k]]
            latencies.append((time.perf_counter() - started) * 1000.0)
            relevant = {
                chunk_id
                for document_id in sample.relevant_documents
                for chunk_id in relevant_by_document.get(document_id, [])
            }
            samples.append((retrieved_ids, relevant))

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
        )


def run_experiment(documents: list[Document], base_config: RagConfig) -> ExperimentResult:
    """Run the experiment declared in ``base_config.experiments``."""
    if base_config.experiments is None:
        raise ValueError("configuration has no 'experiments' section")
    return ExperimentRunner(documents, base_config, base_config.experiments).run()