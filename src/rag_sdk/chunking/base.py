"""Chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from rag_sdk.config import ChunkerConfig
from rag_sdk.core import Chunk, Document


class Chunker(ABC):
    """Splits documents into chunks.

    Implementations must be constructed from a configuration via
    :meth:`from_config` so they can be resolved by name from YAML.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: ChunkerConfig) -> Self:
        """Build a chunker from a validated configuration."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` into ordered, non-empty chunks."""


def build_chunks(document: Document, pieces: list[tuple[str, int, int]]) -> list[Chunk]:
    """Turn ``(text, start, end)`` pieces into :class:`Chunk` objects."""
    return [
        Chunk(
            id=f"{document.id}:{index}",
            text=text,
            document_id=document.id,
            index=index,
            start_char=start,
            end_char=end,
            metadata=document.metadata,
        )
        for index, (text, start, end) in enumerate(pieces)
    ]