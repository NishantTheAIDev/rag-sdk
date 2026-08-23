"""Evaluation pipeline for end-to-end RAG evaluation."""

from __future__ import annotations

from typing import Any

from rag_sdk.config import RagConfig
from rag_sdk.core import Document
from rag_sdk.dataset.loader import load_queries
from rag_sdk.evaluation.answer import EvaluationSample, RAGResult, build_evaluators
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

    def evaluate(self, dataset_path: str) -> dict[str, Any]:
        """Run evaluation on a dataset."""
        queries = load_queries(dataset_path)
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
        for sample in samples:
            # Retrieve
            retrieved = self._retrieval_pipeline.search(sample.query)

            # Build context
            context = self._context_builder.build(retrieved, sample.query)

            # Generate
            generation = None
            if self._generator:
                generation = self._generator.generate(
                    sample.query,
                    context=context.text,
                )

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
            })

        # Aggregate metrics
        metrics = self._aggregate_metrics(all_results)

        return {
            "metrics": metrics,
            "results": all_results,
            "config": self._config.model_dump(mode="json"),
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
) -> dict[str, Any]:
    """Convenience function to run end-to-end RAG evaluation."""
    pipeline = EvaluationPipeline(config, documents)
    return pipeline.evaluate(dataset_path)