"""Answer evaluation for RAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from rag_sdk.generation import GenerationResponse


@dataclass
class EvaluationSample:
    """A single evaluation sample."""

    query_id: str
    query: str
    relevant_documents: list[str]
    relevant_chunks: list[str]
    reference_answer: str | None = None


@dataclass
class RAGResult:
    """Result from a RAG pipeline for evaluation."""

    query: str
    query_id: str
    retrieved_chunks: list[Any]
    generation: GenerationResponse | None = None
    retrieval_metrics: dict[str, float] | None = None


class EvaluationResult(BaseModel):
    """Result of a single evaluation metric."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    score: float
    reason: str | None = None
    evaluator_metadata: dict[str, Any] = {}
    errors: list[str] = []


class Evaluator(Protocol):
    """Protocol for evaluators."""

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        ...


class BaseEvaluator(ABC):
    """Base class for evaluators."""

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name

    @abstractmethod
    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        ...

    def _make_result(self, score: float, reason: str | None = None) -> EvaluationResult:
        return EvaluationResult(metric_name=self.metric_name, score=score, reason=reason)


class FaithfulnessEvaluator(BaseEvaluator):
    """Reference-based faithfulness evaluator using token overlap."""

    def __init__(self) -> None:
        super().__init__("faithfulness")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.generation or not sample.reference_answer:
            return self._make_result(0.0, "No generation or reference answer")

        gen_tokens = set(result.generation.text.lower().split())
        ref_tokens = set(sample.reference_answer.lower().split())

        if not gen_tokens:
            return self._make_result(0.0, "Empty generation")

        overlap = gen_tokens & ref_tokens
        score = len(overlap) / len(gen_tokens)
        return self._make_result(score, f"Token overlap: {len(overlap)}/{len(gen_tokens)}")


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Reference-based answer relevance evaluator."""

    def __init__(self) -> None:
        super().__init__("answer_relevance")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.generation:
            return self._make_result(0.0, "No generation")

        # Simple heuristic: check if query terms appear in answer
        query_tokens = set(sample.query.lower().split())
        gen_tokens = set(result.generation.text.lower().split())

        if not query_tokens:
            return self._make_result(0.0, "Empty query")

        overlap = query_tokens & gen_tokens
        score = len(overlap) / len(query_tokens)
        return self._make_result(score, f"Query token overlap: {len(overlap)}/{len(query_tokens)}")


def not_applicable(metric_name: str, reason: str) -> EvaluationResult:
    """A result for a sample the metric cannot score (e.g. no relevance labels).

    Aggregation skips these instead of counting them as 0 or 1.
    """
    return EvaluationResult(
        metric_name=metric_name,
        score=0.0,
        reason=reason,
        evaluator_metadata={"applicable": False},
    )


def is_applicable(result: EvaluationResult) -> bool:
    """Whether ``result`` should count towards aggregate metrics."""
    return result.evaluator_metadata.get("applicable", True) is not False


def _retrieved_chunks(result: RAGResult) -> list[Any]:
    return [c.chunk for c in result.retrieved_chunks if hasattr(c, "chunk")]


class ContextPrecisionEvaluator(BaseEvaluator):
    """Context precision: fraction of retrieved chunks that are relevant.

    Uses ``relevant_chunks`` when the sample has them; otherwise a retrieved
    chunk is relevant when its document is in ``relevant_documents``. Samples
    with neither are not applicable and are left out of the aggregate.
    """

    def __init__(self) -> None:
        super().__init__("context_precision")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not sample.relevant_chunks and not sample.relevant_documents:
            return not_applicable(self.metric_name, "No relevance labels")
        if not result.retrieved_chunks:
            return self._make_result(0.0, "No retrieved chunks")

        chunks = _retrieved_chunks(result)
        if not chunks:
            return self._make_result(0.0, "No retrieved chunk IDs")

        if sample.relevant_chunks:
            level = "chunk"
            relevant = set(sample.relevant_chunks)
            relevant_retrieved = sum(1 for c in chunks if c.id in relevant)
        else:
            level = "document"
            relevant = set(sample.relevant_documents)
            relevant_retrieved = sum(1 for c in chunks if c.document_id in relevant)
        score = relevant_retrieved / len(chunks)
        return self._make_result(
            score, f"Relevant in top-K ({level} labels): {relevant_retrieved}/{len(chunks)}"
        )


class ContextRecallEvaluator(BaseEvaluator):
    """Context recall: fraction of relevant items that were retrieved.

    With ``relevant_chunks``, the fraction of those chunks retrieved; otherwise
    the fraction of ``relevant_documents`` with at least one retrieved chunk.
    Samples with neither are not applicable and are left out of the aggregate.
    """

    def __init__(self) -> None:
        super().__init__("context_recall")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        chunks = _retrieved_chunks(result)
        if sample.relevant_chunks:
            relevant = set(sample.relevant_chunks)
            found = len(relevant & {c.id for c in chunks})
            return self._make_result(
                found / len(relevant), f"Relevant chunks found: {found}/{len(relevant)}"
            )
        if sample.relevant_documents:
            relevant = set(sample.relevant_documents)
            found = len(relevant & {c.document_id for c in chunks})
            return self._make_result(
                found / len(relevant), f"Relevant documents found: {found}/{len(relevant)}"
            )
        return not_applicable(self.metric_name, "No relevance labels")


class CorrectnessEvaluator(BaseEvaluator):
    """Correctness evaluator using fuzzy matching.

    Samples without a ``reference_answer``, or runs without generation, are
    not applicable and are left out of the aggregate.
    """

    def __init__(self) -> None:
        super().__init__("correctness")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not sample.reference_answer:
            return not_applicable(self.metric_name, "No reference answer")
        if result.generation is None:
            return not_applicable(self.metric_name, "No generation")

        gen = result.generation.text.strip().lower()
        ref = sample.reference_answer.strip().lower()

        if gen == ref:
            return self._make_result(1.0, "Exact match")

        # Simple token-based fuzzy match
        gen_tokens = set(gen.split())
        ref_tokens = set(ref.split())
        if not gen_tokens or not ref_tokens:
            return self._make_result(0.0, "Empty tokens")

        overlap = gen_tokens & ref_tokens
        precision = len(overlap) / len(gen_tokens)
        recall = len(overlap) / len(ref_tokens)
        denom = precision + recall
        score = 0.0 if denom == 0 else 2 * precision * recall / denom  # F1

        return self._make_result(score, f"Token F1: P={precision:.2f}, R={recall:.2f}")


class CitationAccuracyEvaluator(BaseEvaluator):
    """Citation accuracy: do citations point to relevant chunks (or documents)?"""

    def __init__(self) -> None:
        super().__init__("citation_accuracy")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not sample.relevant_chunks and not sample.relevant_documents:
            return not_applicable(self.metric_name, "No relevance labels")
        if not result.generation or not result.generation.cited_answer:
            return self._make_result(0.0, "No citations in generation")

        citations = result.generation.cited_answer.citations
        if not citations:
            return self._make_result(0.0, "No citations")

        # Chunk labels when present, otherwise the cited chunk's document.
        if sample.relevant_chunks:
            relevant = set(sample.relevant_chunks)
            correct = sum(1 for c in citations if c.chunk_id in relevant)
        else:
            relevant = set(sample.relevant_documents)
            correct = sum(1 for c in citations if c.document_id in relevant)
        score = correct / len(citations)
        return self._make_result(score, f"Correct citations: {correct}/{len(citations)}")


# LLM-as-Judge Evaluators

class LLMFaithfulnessEvaluator(BaseEvaluator):
    """LLM-as-judge faithfulness evaluator."""

    def __init__(self, judge_generator) -> None:
        super().__init__("faithfulness_llm")
        self._judge = judge_generator

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.generation:
            return self._make_result(0.0, "No generation")

        chunks_with_text = [
            c.chunk.text for c in result.retrieved_chunks if hasattr(c, "chunk")
        ]
        context_text = "\n".join(chunks_with_text)

        prompt = f"""Task: Evaluate if the answer is faithful to the provided context.

Context:
{context_text}

Answer:
{result.generation.text}

Is the answer faithful to the context? Answer with a score from 0.0 to 1.0 and a brief reason.
Format: SCORE: <float> REASON: <text>"""

        try:
            response = self._judge.generate(prompt)
            score = self._parse_score(response.text)
            return self._make_result(score, response.text)
        except Exception as e:
            return self._make_result(0.0, f"Judge error: {e}")

    def _parse_score(self, text: str) -> float:
        import re
        match = re.search(r"SCORE:\s*([0-9.]+)", text)
        if match:
            return float(match.group(1))
        return 0.0


class LLMAnswerRelevanceEvaluator(BaseEvaluator):
    """LLM-as-judge answer relevance evaluator."""

    def __init__(self, judge_generator) -> None:
        super().__init__("answer_relevance_llm")
        self._judge = judge_generator

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.generation:
            return self._make_result(0.0, "No generation")

        prompt = f"""Task: Evaluate if the answer addresses the query.

Query:
{sample.query}

Answer:
{result.generation.text}

Does the answer address the query? Answer with a score from 0.0 to 1.0 and a brief reason.
Format: SCORE: <float> REASON: <text>"""

        try:
            response = self._judge.generate(prompt)
            score = self._parse_score(response.text)
            return self._make_result(score, response.text)
        except Exception as e:
            return self._make_result(0.0, f"Judge error: {e}")

    def _parse_score(self, text: str) -> float:
        import re
        match = re.search(r"SCORE:\s*([0-9.]+)", text)
        if match:
            return float(match.group(1))
        return 0.0


class LLMCorrectnessEvaluator(BaseEvaluator):
    """LLM-as-judge correctness evaluator."""

    def __init__(self, judge_generator) -> None:
        super().__init__("correctness_llm")
        self._judge = judge_generator

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.generation or not sample.reference_answer:
            return self._make_result(0.0, "No generation or reference answer")

        prompt = f"""Task: Evaluate if the answer is correct compared to the reference.

Reference Answer:
{sample.reference_answer}

Generated Answer:
{result.generation.text}

Is the generated answer correct? Answer with a score from 0.0 to 1.0 and a brief reason.
Format: SCORE: <float> REASON: <text>"""

        try:
            response = self._judge.generate(prompt)
            score = self._parse_score(response.text)
            return self._make_result(score, response.text)
        except Exception as e:
            return self._make_result(0.0, f"Judge error: {e}")

    def _parse_score(self, text: str) -> float:
        import re
        match = re.search(r"SCORE:\s*([0-9.]+)", text)
        if match:
            return float(match.group(1))
        return 0.0


def build_evaluators(config, judge_generator=None) -> list[Evaluator]:
    """Build evaluators from configuration."""
    evaluators = []

    if config.reference_based:
        evaluators.extend([
            FaithfulnessEvaluator(),
            AnswerRelevanceEvaluator(),
            ContextPrecisionEvaluator(),
            ContextRecallEvaluator(),
            CorrectnessEvaluator(),
            CitationAccuracyEvaluator(),
        ])

    if config.llm_judge and judge_generator:
        evaluators.extend([
            LLMFaithfulnessEvaluator(judge_generator),
            LLMAnswerRelevanceEvaluator(judge_generator),
            LLMCorrectnessEvaluator(judge_generator),
        ])

    # Filter by configured metrics
    if config.metrics:
        evaluators = [e for e in evaluators if e.metric_name in config.metrics]

    return evaluators