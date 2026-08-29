"""Vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class VectorStore(ABC):
    """Stores vectors and returns nearest neighbors for a query vector."""

    @abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Add ``vectors`` (``(n, dim)`` float32) associated with ``ids``."""

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        k: int,
        filters: dict[str, str | int | float | bool | list[str] | list[int]] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, score)`` pairs, highest score first.

        Args:
            vector: Query vector.
            k: Number of results to return.
            filters: Optional metadata filters (e.g., {"category": "finance", "year": 2024}).
        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of vectors stored."""

    @abstractmethod
    def get_embedding(self, chunk_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a chunk ID.

        Returns None if the chunk ID is not found.
        """