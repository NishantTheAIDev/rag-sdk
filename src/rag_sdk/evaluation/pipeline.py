"""Evaluation pipeline for end-to-end RAG evaluation."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rag_sdk.config import RagConfig
from rag_sdk.core import Document
from rag_sdk.dataset.loader import load_queries
from rag_sdk.evaluation.answer import EvaluationSample, RAGResult, build_evaluators
from rag_sdk.evaluation.relevance import judge_retrieval
from rag_sdk.evaluation.retrieval_metrics import evaluate_retrieval
from rag_sdk.generation import build_generator
from rag_sdk.indexing import FaissVectorStore, InMemoryChunkStore, InMemoryDocumentStore
from rag_sdk.ingestion import IngestionResult, ingest_documents
from rag_sdk.retrieval import build_retrieval_pipeline


class EvaluationPipeline:
    """End-to-end evaluation pipeline."""

    def __init__(self, config: RagConfig, documents: list[Document]) -> None:
        self._config = config
        self._documents = documents

        # Build components
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build all pipeline components from config."""
        from rag_sdk.chunking import build_chunker
        from rag_sdk.embeddings import build_embedding_provider

        chunker = build_chunker(self._config.chunking)
        embedding = build_embedding_provider(self._config.embedding)
        store = FaissVectorStore(dimension=embedding.dimension)
        chunk_store = InMemoryChunkStore()
        document_store = InMemoryDocumentStore()

        # Ingest
        self._ingestion_result: IngestionResult = ingest_documents(
            self._documents,
            chunker,
            embedding,
            store,
            chunk_store=chunk_store,
            document_store=document_store,
            preprocessing_config=self._config.preprocessing,
        )

        # Build retrieval pipeline
        self._retrieval_pipeline = build_retrieval_pipeline(
            self._config,
            embedding,
            store,
            chunk_store=chunk_store,
            document_store=document_store,
            chunks=self._ingestion_result.all_chunks,
        )

        # Build generator
        self._generator = None
        if self._config.generation:
            self._generator = build_generator(self._config.generation)

        # Build judge generator for LLM evaluators
        self._judge_generator = None
        if self._config.evaluation and self._config.evaluation.answer.llm_judge:
            judge_config = self._config.evaluation.answer.llm_judge
            from rag_sdk.generation.base import MockGeneratorConfig
            if judge_config.provider == "mock":
                judge_config = MockGeneratorConfig(
                    provider="mock",
                    model="mock-judge",
                    temperature=judge_config.temperature,
                )
            self._judge_generator = build_generator(judge_config)

        # Build context builder
        from rag_sdk.context import ContextConfig, build_context_builder
        from rag_sdk.tokenizer.base import Cl100kBaseTokenizerConfig, WhitespaceTokenizerConfig
        if self._config.retrieval:
            # Get tokenizer from auto_merging config if available
            tokenizer_str = None
            if hasattr(self._config.retrieval, "auto_merging"):
                tokenizer_str = self._config.retrieval.auto_merging.tokenizer
            tokenizer = None
            if tokenizer_str == "whitespace":
                tokenizer = WhitespaceTokenizerConfig()
            elif tokenizer_str == "cl100k_base":
                tokenizer = Cl100kBaseTokenizerConfig()
            context_config = ContextConfig(
                max_tokens=self._config.retrieval.top_k * 256,  # rough estimate
                tokenizer=tokenizer,
            )
        else:
            context_config = ContextConfig()
        self._context_builder = build_context_builder(context_config)

        # Build evaluators
        self._evaluators = []
        if self._config.evaluation:
            self._evaluators = build_evaluators(
                self._config.evaluation.answer,
                self._judge_generator,
            )

    def evaluate(
        self,
        dataset_path: str,
        *,
        k: int | None = None,
        relevance_level: Literal["document", "chunk"] | None = None,
    ) -> dict[str, Any]:
        """Run retrieval, generation and answer evaluation on a dataset.

        Retrieval and answer quality are reported separately:
        ``retrieval_metrics`` (Hit/Recall/Precision@k, MRR, nDCG, MAP) and
        ``metrics`` (answer evaluators; also available as ``answer_metrics``).
        ``k`` defaults to ``evaluation.k`` then ``retrieval.top_k``;
        ``relevance_level`` defaults to ``evaluation.relevance_level``.
        """
        evaluation = self._config.evaluation
        if k is None:
            k = (evaluation.k if evaluation else None) or self._config.retrieval.top_k
        if relevance_level is None:
            relevance_level = evaluation.relevance_level if evaluation else "document"
        queries = load_queries(dataset_path)
        chunks_by_document: dict[str, list[str]] = {}
        for chunk in self._ingestion_result.all_chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk.id)
        retrieval_samples = []
        retrieval_latencies: list[float] = []
        total_latencies: list[float] = []
        samples = [
            EvaluationSample(
                query_id=q.query_id,
                query=q.query,
                relevant_documents=q.relevant_documents,
                relevant_chunks=q.relevant_chunks,
                reference_answer=q.reference_answer,
            )
            for q in queries
        ]

        all_results = []
        for query, sample in zip(queries, samples, strict=True):
            # Retrieve
            started = time.perf_counter()
            retrieved = self._retrieval_pipeline.search(sample.query)
            retrieval_ms = (time.perf_counter() - started) * 1000.0
            retrieval_latencies.append(retrieval_ms)
            retrieval_samples.append(
                judge_retrieval(retrieved[:k], query, chunks_by_document, relevance_level)
            )

            # Build context
            context = self._context_builder.build(retrieved, sample.query)

            # Generate
            generation = None
            if self._generator:
                generation = self._generator.generate(
                    sample.query,
                    context=context.text,
                )
            total_ms = (time.perf_counter() - started) * 1000.0
            total_latencies.append(total_ms)

            # Build RAG result
            rag_result = RAGResult(
                query=sample.query,
                query_id=sample.query_id,
                retrieved_chunks=retrieved,
                generation=generation,
            )

            # Evaluate
            eval_results = []
            for evaluator in self._evaluators:
                eval_results.append(evaluator.evaluate(sample, rag_result))

            all_results.append({
                "sample": sample,
                "rag_result": rag_result,
                "eval_results": eval_results,
                "context": context,
                "latency_ms": {"retrieval": retrieval_ms, "total": total_ms},
            })

        # Aggregate metrics
        metrics = self._aggregate_metrics(all_results)

        return {
            "metrics": metrics,
            "answer_metrics": metrics,
            "retrieval_metrics": evaluate_retrieval(retrieval_samples, k=k),
            "k": k,
            "relevance_level": relevance_level,
            "latency_ms": {
                "retrieval_mean_ms": statistics.fmean(retrieval_latencies),
                "retrieval_median_ms": statistics.median(retrieval_latencies),
                "total_mean_ms": statistics.fmean(total_latencies),
                "total_median_ms": statistics.median(total_latencies),
            },
            "dataset": {
                "path": str(dataset_path),
                "hash": _hash_file(Path(dataset_path)),
                "queries": len(queries),
            },
            "models": self._model_info(),
            "timestamp": datetime.now(UTC).isoformat(),
            "total_chunks": self._ingestion_result.total_chunks,
            "results": all_results,
            "config": self._config.model_dump(mode="json"),
        }

    def _model_info(self) -> dict[str, str | None]:
        config = self._config
        reranker = config.reranker
        judge = config.evaluation.answer.llm_judge if config.evaluation else None
        return {
            "embedding": f"{config.embedding.provider}:{config.embedding.model or 'default'}",
            "reranker": (
                None
                if reranker is None or reranker.strategy == "none"
                else f"{reranker.strategy}:{getattr(reranker, 'model', None)}"
            ),
            "generator": (
                f"{config.generation.provider}:{config.generation.model}"
                if config.generation
                else None
            ),
            "judge": f"{judge.provider}:{judge.model}" if judge else None,
        }

    def _aggregate_metrics(self, results: list[dict]) -> dict[str, float]:
        """Aggregate evaluation metrics across all samples."""
        from collections import defaultdict

        metric_sums = defaultdict(float)
        metric_counts = defaultdict(int)

        for r in results:
            for eval_result in r["eval_results"]:
                metric_sums[eval_result.metric_name] += eval_result.score
                metric_counts[eval_result.metric_name] += 1

        return {
            name: metric_sums[name] / metric_counts[name]
            for name in metric_sums
        }


def evaluate_rag(
    config: RagConfig,
    documents: list[Document],
    dataset_path: str,
    *,
    k: int | None = None,
    relevance_level: Literal["document", "chunk"] | None = None,
) -> dict[str, Any]:
    """Convenience function to run end-to-end RAG evaluation."""
    pipeline = EvaluationPipeline(config, documents)
    return pipeline.evaluate(dataset_path, k=k, relevance_level=relevance_level)


def evaluation_report(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable summary of an :func:`evaluate_rag` result.

    Records everything needed to reproduce and compare the run: configuration,
    dataset path/hash, retrieval and answer metrics, latency, timestamp and
    model information, plus a per-query breakdown.
    """
    queries = []
    for item in result["results"]:
        sample: EvaluationSample = item["sample"]
        rag_result: RAGResult = item["rag_result"]
        queries.append({
            "query_id": sample.query_id,
            "query": sample.query,
            "relevant_documents": sample.relevant_documents,
            "retrieved": [
                {"chunk_id": r.chunk.id, "document_id": r.chunk.document_id, "score": r.score}
                for r in rag_result.retrieved_chunks
            ],
            "answer": rag_result.generation.text if rag_result.generation else None,
            "reference_answer": sample.reference_answer,
            "answer_scores": {
                e.metric_name: e.score for e in item["eval_results"]
            },
            "latency_ms": item.get("latency_ms"),
        })
    return {
        "timestamp": result["timestamp"],
        "dataset": result["dataset"],
        "models": result["models"],
        "k": result["k"],
        "relevance_level": result["relevance_level"],
        "retrieval_metrics": result["retrieval_metrics"],
        "answer_metrics": result["answer_metrics"],
        "latency_ms": result["latency_ms"],
        "total_chunks": result["total_chunks"],
        "config": result["config"],
        "queries": queries,
    }


def write_evaluation_report(result: dict[str, Any], path: str | Path) -> Path:
    """Write :func:`evaluation_report` as JSON to ``path``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evaluation_report(result), indent=2, default=str), encoding="utf-8"
    )
    return output


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()