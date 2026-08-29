"""Sentence Transformers embedding provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rag_sdk.config import EmbeddingConfig
from rag_sdk.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers library."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


def build_sentence_transformers_provider(
    config: EmbeddingConfig,
) -> SentenceTransformersEmbeddingProvider:
    """Build a sentence-transformers embedding provider from config."""
    model = config.model or "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformersEmbeddingProvider(model)