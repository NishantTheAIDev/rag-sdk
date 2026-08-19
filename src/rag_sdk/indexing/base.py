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
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, score)`` pairs, highest score first."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of vectors stored."""