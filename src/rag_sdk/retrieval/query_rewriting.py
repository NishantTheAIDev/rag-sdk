"""Query rewriting for retrieval enhancement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rag_sdk.config import QueryRewriterConfig
from rag_sdk.generation import Generator

if TYPE_CHECKING:
    pass


class QueryRewriter(ABC):
    """Abstract base class for query rewriters."""

    @abstractmethod
    def rewrite(self, query: str, context: dict | None = None) -> str:
        """Rewrite a query for better retrieval."""
        pass


class LLMQueryRewriter(QueryRewriter):
    """Uses an LLM to rewrite queries (e.g., query expansion, HyDE)."""

    def __init__(self, generator: Generator, prompt: str | None = None):
        self._generator = generator
        self._prompt = prompt or (
            "Rewrite the following query to be more effective for document retrieval. "
            "Focus on key terms and concepts. Return only the rewritten query.\n\n"
            "Original query: {query}\n\nRewritten query:"
        )

    def rewrite(self, query: str, context: dict | None = None) -> str:
        formatted = self._prompt.format(query=query)
        response = self._generator.generate(formatted, [])
        return response.text.strip()


class TemplateQueryRewriter(QueryRewriter):
    """Applies template-based transformations to queries."""

    def __init__(self, template: str):
        self._template = template

    def rewrite(self, query: str, context: dict | None = None) -> str:
        return self._template.format(query=query)


class HyDEQueryRewriter(QueryRewriter):
    """Hypothetical Document Embeddings (HyDE) query rewriter.

    Generates a hypothetical document that would answer the query,
    then uses that document as the query for retrieval.
    """

    def __init__(self, generator: Generator, prompt: str | None = None):
        self._generator = generator
        self._prompt = prompt or (
            "Write a hypothetical document that would perfectly answer the following query. "
            "The document should be detailed and contain relevant information.\n\n"
            "Query: {query}\n\nHypothetical document:"
        )

    def rewrite(self, query: str, context: dict | None = None) -> str:
        formatted = self._prompt.format(query=query)
        response = self._generator.generate(formatted, [])
        return response.text.strip()


class QueryRewriterFactory:
    """Factory for creating query rewriters from config."""

    @staticmethod
    def create(
        config: QueryRewriterConfig, generator: Generator | None = None
    ) -> QueryRewriter | None:
        if not config.enabled:
            return None

        if config.strategy == "llm":
            if not generator:
                raise ValueError("LLM query rewriter requires a generator")
            return LLMQueryRewriter(generator, config.prompt)

        elif config.strategy == "template":
            if not config.template:
                raise ValueError("Template query rewriter requires a template")
            return TemplateQueryRewriter(config.template)

        elif config.strategy == "hyde":
            if not generator:
                raise ValueError("HyDE query rewriter requires a generator")
            return HyDEQueryRewriter(generator, config.prompt)

        return None