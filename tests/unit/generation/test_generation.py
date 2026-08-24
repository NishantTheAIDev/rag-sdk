"""Tests for generation providers."""

from __future__ import annotations

from rag_sdk.generation import (
    Citation,
    CitedAnswer,
    GenerationResponse,
    MockGenerator,
    MockGeneratorConfig,
)


def test_mock_generator_basic():
    """Test mock generator basic functionality."""
    config = MockGeneratorConfig(model="mock", canned_response="Test response")
    generator = MockGenerator(config)

    response = generator.generate("Test prompt")

    assert isinstance(response, GenerationResponse)
    assert response.text == "Test response"
    assert response.model == "mock"
    assert response.provider == "mock"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens


def test_mock_generator_with_context():
    """Test mock generator with context."""
    config = MockGeneratorConfig(model="mock", canned_response="Response with context")
    generator = MockGenerator(config)

    response = generator.generate("Test prompt", context="Some context")

    assert response.text == "Response with context"


def test_mock_generator_model_name():
    """Test mock generator model_name property."""
    config = MockGeneratorConfig(model="test-model")
    generator = MockGenerator(config)

    assert generator.model_name == "test-model"
    assert generator.provider_name == "mock"


def test_generation_response_cited_answer():
    """Test GenerationResponse with cited answer."""
    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        text_span=(0, 10),
        score=0.9,
    )
    cited_answer = CitedAnswer(text="Answer", citations=[citation])

    response = GenerationResponse(
        text="Answer",
        cited_answer=cited_answer,
        prompt_tokens=10,
        completion_tokens=5,
    )

    assert response.cited_answer is not None
    assert len(response.cited_answer.citations) == 1
    assert response.cited_answer.citations[0].document_id == "doc1"


def test_citation_fields():
    """Test citation with all fields."""
    citation = Citation(
        document_id="doc1",
        chunk_id="chunk1",
        page=5,
        section="Introduction",
        source_uri="file:///doc1.pdf",
        text_span=(100, 200),
        score=0.95,
    )

    assert citation.document_id == "doc1"
    assert citation.page == 5
    assert citation.section == "Introduction"
    assert citation.source_uri == "file:///doc1.pdf"
    assert citation.text_span == (100, 200)
    assert citation.score == 0.95


def test_mock_generator_deterministic():
    """Test mock generator returns same response for same config."""
    config = MockGeneratorConfig(model="mock", canned_response="Fixed response")
    generator = MockGenerator(config)

    response1 = generator.generate("Prompt 1")
    response2 = generator.generate("Prompt 2")

    assert response1.text == response2.text == "Fixed response"