"""Context builder base classes and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag_sdk.retrieval.base import RetrievalResult
from rag_sdk.tokenizer import Tokenizer, TokenizerConfig
from rag_sdk.tokenizer.base import WhitespaceTokenizerConfig


@dataclass
class Context:
    """Constructed context for generation."""

    text: str
    token_count: int
    source_chunks: list[RetrievalResult]
    tokenizer_type: str


class ContextConfig(BaseModel):
    """Configuration for context construction."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=2048, ge=1)
    include_metadata: bool = True
    citation_format: str = "[{doc_id}:{chunk_id}]"
    deduplicate: bool = True
    tokenizer: WhitespaceTokenizerConfig = Field(default_factory=WhitespaceTokenizerConfig)


class ContextBuilder(Protocol):
    """Protocol for context builders."""

    def build(
        self,
        retrieved_chunks: list[RetrievalResult],
        query: str,
        config: ContextConfig,
    ) -> Context:
        """Build context from retrieved chunks."""
        ...


class ContextBuilderBase:
    """Base class for context builders with common utilities."""

    def __init__(self, config: ContextConfig) -> None:
        self._config = config
        self._tokenizer = self._create_tokenizer(config.tokenizer)

    def _create_tokenizer(self, tokenizer_config: TokenizerConfig) -> Tokenizer:
        from rag_sdk.tokenizer import build_tokenizer

        return build_tokenizer(tokenizer_config)

    def _count_tokens(self, text: str) -> int:
        return self._tokenizer.count(text)

    def _format_chunk(self, chunk: RetrievalResult, index: int) -> str:
        parts = []
        if self._config.include_metadata:
            doc_id = chunk.chunk.document_id
            chunk_id = chunk.chunk.id
            parts.append(f"[{doc_id}:{chunk_id}]")
        parts.append(chunk.chunk.text)
        return " ".join(parts)

    def _deduplicate_chunks(self, chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        if not self._config.deduplicate:
            return chunks

        seen = set()
        unique = []
        for chunk in chunks:
            if chunk.chunk.id not in seen:
                seen.add(chunk.chunk.id)
                unique.append(chunk)
        return unique

    def _apply_token_budget(self, chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        """Apply token budget to chunks, keeping highest-ranked ones."""
        if not chunks:
            return chunks

        total = 0
        result = []
        for chunk in chunks:
            chunk_tokens = self._count_tokens(self._format_chunk(chunk, len(result)))
            if total + chunk_tokens > self._config.max_tokens and result:
                break
            total += chunk_tokens
            result.append(chunk)
        return result