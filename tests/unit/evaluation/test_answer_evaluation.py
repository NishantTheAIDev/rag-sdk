"""Tests for answer evaluation."""

from __future__ import annotations

from rag_sdk.core import Chunk
from rag_sdk.evaluation.answer import (
    AnswerRelevanceEvaluator,
    CitationAccuracyEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    CorrectnessEvaluator,
    EvaluationSample,
    FaithfulnessEvaluator,
    RAGResult,
)
from rag_sdk.generation import Citation, CitedAnswer, GenerationResponse
from rag_sdk.retrieval.base import RetrievalResult


def _make_chunk(text: str, chunk_id: str = "c1") -> RetrievalResult:
    chunk = Chunk(
        id=chunk_id,
        document_id="d1",
        text=text,
        index=0,
        start_char=0,
        end_char=len(text),
    )
    return RetrievalResult(query="test", chunk=chunk, score=1.0)


def test_faithfulness_evaluator():
    """Test faithfulness evaluator."""
    evaluator = FaithfulnessEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="What is the capital?",
        relevant_documents=["d1"],
        relevant_chunks=["c1"],
        reference_answer="Paris is the capital of France.",
    )
    generation = GenerationResponse(text="Paris is the capital of France.")
    result = RAGResult(
        query="What is the capital?",
        query_id="q1",
        retrieved_chunks=[],
        generation=generation,
    )

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "faithfulness"
    assert eval_result.score == 1.0


def test_faithfulness_evaluator_partial():
    """Test faithfulness with partial overlap."""
    evaluator = FaithfulnessEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="What is the capital?",
        relevant_documents=["d1"],
        relevant_chunks=["c1"],
        reference_answer="Paris is the capital of France.",
    )
    generation = GenerationResponse(text="Paris is a city.")
    result = RAGResult(
        query="What is the capital?",
        query_id="q1",
        retrieved_chunks=[],
        generation=generation,
    )

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "faithfulness"
    assert 0.0 < eval_result.score < 1.0


def test_answer_relevance_evaluator():
    """Test answer relevance evaluator."""
    evaluator = AnswerRelevanceEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="What is the capital of France?",
        relevant_documents=["d1"],
        relevant_chunks=["c1"],
    )
    generation = GenerationResponse(text="The capital of France is Paris.")
    result = RAGResult(
        query="What is the capital of France?",
        query_id="q1",
        retrieved_chunks=[],
        generation=generation,
    )

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "answer_relevance"
    assert eval_result.score > 0.0


def test_context_precision_evaluator():
    """Test context precision evaluator."""
    evaluator = ContextPrecisionEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="test",
        relevant_documents=["d1"],
        relevant_chunks=["c1", "c2"],
    )
    retrieved = [_make_chunk("Relevant", "c1"), _make_chunk("Not relevant", "c3")]
    result = RAGResult(query="test", query_id="q1", retrieved_chunks=retrieved)

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "context_precision"
    assert eval_result.score == 0.5


def test_context_recall_evaluator():
    """Test context recall evaluator."""
    evaluator = ContextRecallEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="test",
        relevant_documents=["d1"],
        relevant_chunks=["c1", "c2", "c3"],
    )
    retrieved = [_make_chunk("Relevant 1", "c1"), _make_chunk("Relevant 2", "c2")]
    result = RAGResult(query="test", query_id="q1", retrieved_chunks=retrieved)

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "context_recall"
    assert eval_result.score == 2 / 3


def test_correctness_evaluator():
    """Test correctness evaluator."""
    evaluator = CorrectnessEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="test",
        relevant_documents=["d1"],
        relevant_chunks=["c1"],
        reference_answer="The answer is 42.",
    )
    generation = GenerationResponse(text="The answer is 42.")
    result = RAGResult(query="test", query_id="q1", retrieved_chunks=[], generation=generation)

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "correctness"
    assert eval_result.score == 1.0


def test_citation_accuracy_evaluator():
    """Test citation accuracy evaluator."""
    evaluator = CitationAccuracyEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="test",
        relevant_documents=["d1"],
        relevant_chunks=["c1", "c2"],
    )
    citation = Citation(document_id="d1", chunk_id="c1", score=0.9)
    cited_answer = CitedAnswer(text="Answer", citations=[citation])
    generation = GenerationResponse(text="Answer", cited_answer=cited_answer)
    result = RAGResult(query="test", query_id="q1", retrieved_chunks=[], generation=generation)

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "citation_accuracy"
    assert eval_result.score == 1.0


def test_citation_accuracy_evaluator_wrong():
    """Test citation accuracy with wrong citation."""
    evaluator = CitationAccuracyEvaluator()
    sample = EvaluationSample(
        query_id="q1",
        query="test",
        relevant_documents=["d1"],
        relevant_chunks=["c1"],
    )
    citation = Citation(document_id="d1", chunk_id="c999", score=0.9)
    cited_answer = CitedAnswer(text="Answer", citations=[citation])
    generation = GenerationResponse(text="Answer", cited_answer=cited_answer)
    result = RAGResult(query="test", query_id="q1", retrieved_chunks=[], generation=generation)

    eval_result = evaluator.evaluate(sample, result)

    assert eval_result.metric_name == "citation_accuracy"
    assert eval_result.score == 0.0