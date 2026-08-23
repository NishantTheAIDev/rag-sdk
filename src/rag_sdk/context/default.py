"""Default context builder implementation."""

from __future__ import annotations

from rag_sdk.context.base import Context, ContextBuilderBase, ContextConfig
from rag_sdk.retrieval.base import RetrievalResult


class DefaultContextBuilder(ContextBuilderBase):
    """Default context builder with token budgeting and deduplication."""

    def build(
        self,
        retrieved_chunks: list[RetrievalResult],
        query: str,
        config: ContextConfig | None = None,
    ) -> Context:
        """Build context from retrieved chunks."""
        cfg = config or self._config
        tokenizer = self._create_tokenizer(cfg.tokenizer)

        # Deduplicate
        chunks = self._deduplicate_chunks(retrieved_chunks)

        # Apply token budget
        chunks = self._apply_token_budget(chunks)

        # Format context text
        parts = []
        for i, chunk in enumerate(chunks):
            formatted = self._format_chunk(chunk, i)
            parts.append(formatted)

        text = "\n\n".join(parts)
        token_count = tokenizer.count(text)

        return Context(
            text=text,
            token_count=token_count,
            source_chunks=chunks,
            tokenizer_type=cfg.tokenizer.type,
        )