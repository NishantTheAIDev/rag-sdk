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


class ContextPrecisionEvaluator(BaseEvaluator):
    """Context precision: fraction of retrieved chunks that are relevant."""

    def __init__(self) -> None:
        super().__init__("context_precision")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.retrieved_chunks:
            return self._make_result(0.0, "No retrieved chunks")

        relevant = set(sample.relevant_chunks)
        retrieved = [c.chunk.id for c in result.retrieved_chunks if hasattr(c, "chunk")]

        if not retrieved:
            return self._make_result(0.0, "No retrieved chunk IDs")

        relevant_retrieved = sum(1 for r in retrieved if r in relevant)
        score = relevant_retrieved / len(retrieved)
        return self._make_result(score, f"Relevant in top-K: {relevant_retrieved}/{len(retrieved)}")


class ContextRecallEvaluator(BaseEvaluator):
    """Context recall: fraction of relevant chunks that were retrieved."""

    def __init__(self) -> None:
        super().__init__("context_recall")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not sample.relevant_chunks:
            return self._make_result(1.0, "No relevant chunks to recall")

        relevant = set(sample.relevant_chunks)
        retrieved = set()
        for c in result.retrieved_chunks:
            if hasattr(c, "chunk"):
                retrieved.add(c.chunk.id)

        if not relevant:
            return self._make_result(1.0, "No relevant chunks")

        found = sum(1 for r in relevant if r in retrieved)
        score = found / len(relevant)
        return self._make_result(score, f"Relevant chunks found: {found}/{len(relevant)}")


class CorrectnessEvaluator(BaseEvaluator):
    """Correctness evaluator using fuzzy matching."""

    def __init__(self) -> None:
        super().__init__("correctness")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.generation or not sample.reference_answer:
            return self._make_result(0.0, "No generation or reference answer")

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
    """Citation accuracy: do citations point to relevant chunks?"""

    def __init__(self) -> None:
        super().__init__("citation_accuracy")

    def evaluate(self, sample: EvaluationSample, result: RAGResult) -> EvaluationResult:
        if not result.generation or not result.generation.cited_answer:
            return self._make_result(0.0, "No citations in generation")

        relevant = set(sample.relevant_chunks)
        citations = result.generation.cited_answer.citations

        if not citations:
            return self._make_result(0.0, "No citations")

        correct = sum(1 for c in citations if c.chunk_id in relevant)
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