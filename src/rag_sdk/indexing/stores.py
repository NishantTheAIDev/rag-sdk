"""Document and chunk storage abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_sdk.core import Chunk, Document


class DocumentStore(ABC):
    """Abstract document store for retrieving full document text."""

    @abstractmethod
    def add_document(self, document: Document) -> None:
        """Store a document."""
        ...

    @abstractmethod
    def get_document(self, document_id: str) -> Document | None:
        """Retrieve a document by ID."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the store."""
        ...


class ChunkStore(ABC):
    """Abstract chunk store for parent chunk persistence."""

    @abstractmethod
    def add_parent_chunks(self, chunks: list[Chunk]) -> None:
        """Store parent chunks."""
        ...

    @abstractmethod
    def get_parent_chunks(self, parent_ids: list[str]) -> list[Chunk]:
        """Retrieve parent chunks by IDs."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the store."""
        ...