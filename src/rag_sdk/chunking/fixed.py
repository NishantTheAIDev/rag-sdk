"""Fixed-token chunker."""

from __future__ import annotations

from typing import Self

from rag_sdk.chunking.base import Chunker, build_chunks
from rag_sdk.chunking.text import whitespace_tokens
from rag_sdk.config import ChunkerConfig, FixedTokenChunkerConfig
from rag_sdk.core import Chunk, Document


class FixedTokenChunker(Chunker):
    """Split text into chunks of exactly ``chunk_size`` tokens with token overlap."""

    def __init__(self, config: FixedTokenChunkerConfig) -> None:
        if config.tokenizer != "whitespace":
            raise ValueError(f"Unsupported tokenizer: {config.tokenizer!r}")
        self._chunk_size = config.chunk_size
        self._overlap = config.overlap

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> Self:
        return cls(FixedTokenChunkerConfig.model_validate(config.model_dump()))

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = whitespace_tokens(document.text)
        step = self._chunk_size - self._overlap
        pieces = []
        for start_index in range(0, len(tokens), step):
            group = tokens[start_index : start_index + self._chunk_size]
            if not group:
                break
            start = group[0][1]
            end = group[-1][2]
            pieces.append((document.text[start:end], start, end))
        return build_chunks(document, pieces)