"""Generation engine for RAG answer generation."""

from __future__ import annotations

from rag_sdk.generation.anthropic import AnthropicGenerator
from rag_sdk.generation.base import (
    AnthropicGeneratorConfig,
    Citation,
    CitedAnswer,
    GenerationConfig,
    GenerationResponse,
    Generator,
    MockGeneratorConfig,
    OllamaGeneratorConfig,
    OpenAIGeneratorConfig,
)
from rag_sdk.generation.factory import (
    build_generator,
    generator_registry,
    register_generator,
)
from rag_sdk.generation.mock import MockGenerator
from rag_sdk.generation.ollama import OllamaGenerator
from rag_sdk.generation.openai import OpenAIGenerator

__all__ = [
    "Generator",
    "GenerationConfig",
    "GenerationResponse",
    "CitedAnswer",
    "Citation",
    "MockGeneratorConfig",
    "OpenAIGeneratorConfig",
    "AnthropicGeneratorConfig",
    "OllamaGeneratorConfig",
    "MockGenerator",
    "OpenAIGenerator",
    "AnthropicGenerator",
    "OllamaGenerator",
    "build_generator",
    "generator_registry",
    "register_generator",
]