"""Generator factory resolved by configuration."""

from __future__ import annotations

from rag_sdk.core import Registry
from rag_sdk.generation import GenerationConfig, Generator
from rag_sdk.generation.anthropic import AnthropicGenerator
from rag_sdk.generation.mock import MockGenerator
from rag_sdk.generation.ollama import OllamaGenerator
from rag_sdk.generation.openai import OpenAIGenerator

generator_registry: Registry[type[Generator]] = Registry()
generator_registry.register("mock", MockGenerator)
generator_registry.register("openai", OpenAIGenerator)
generator_registry.register("anthropic", AnthropicGenerator)
generator_registry.register("ollama", OllamaGenerator)


def build_generator(config: GenerationConfig) -> Generator:
    """Construct a generator from its configuration."""
    generator_class = generator_registry.get(config.provider)
    if generator_class is None:
        raise ValueError(f"Unknown generator provider: {config.provider!r}")
    return generator_class(config)


def register_generator(name: str, generator_class: type[Generator]) -> None:
    """Register a custom generator."""
    generator_registry.register(name, generator_class)