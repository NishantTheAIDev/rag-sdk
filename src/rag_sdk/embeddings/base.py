"""Embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class EmbeddingProvider(ABC):
    """Embeds texts into dense vector representations.

    Providers must return a float32 matrix of shape ``(len(texts), dim)``.
    Implementations should normalize vectors so that inner-product search
    corresponds to cosine similarity.
    """

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into a ``(len(texts), dim)`` float32 matrix."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced embeddings."""