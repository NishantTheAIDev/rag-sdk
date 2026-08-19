"""Recursive character-splitting chunker."""

from __future__ import annotations

from typing import Self

from rag_sdk.chunking.base import Chunker, build_chunks
from rag_sdk.chunking.text import apply_overlap, recursive_split
from rag_sdk.config import ChunkerConfig, RecursiveChunkerConfig
from rag_sdk.core import Chunk, Document


class RecursiveChunker(Chunker):
    """Split text hierarchically on separators until chunks fit ``chunk_size``."""

    def __init__(self, config: RecursiveChunkerConfig) -> None:
        self._chunk_size = config.chunk_size
        self._overlap = config.overlap
        self._separators = config.separators

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> Self:
        return cls(RecursiveChunkerConfig.model_validate(config.model_dump()))

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = recursive_split(document.text, self._separators, self._chunk_size)
        pieces = apply_overlap(pieces, self._overlap)
        return build_chunks(document, pieces)