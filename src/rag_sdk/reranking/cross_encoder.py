"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.config import CrossEncoderRerankerConfig
from rag_sdk.reranking.base import RerankerProvider
from rag_sdk.retrieval.base import RetrievalResult


class CrossEncoderReranker(RerankerProvider):
    """Reranker using a cross-encoder model from sentence-transformers."""

    def __init__(self, model: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with 'pip install sentence-transformers'."
            ) from e

        self._model = CrossEncoder(model, device=device)
        self._model_name = model

    @classmethod
    def from_config(cls, config: CrossEncoderRerankerConfig) -> CrossEncoderReranker:
        return cls(model=config.model, device=config.device)

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, c.chunk.text) for c in candidates]

        # Get scores
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Sort by score descending
        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build reranked results
        reranked = []
        for rank, (candidate, score) in enumerate(scored[:top_k]):
            new_result = candidate.model_copy(deep=True)
            new_result.rerank_score = float(score)
            new_result.rerank_rank = rank + 1
            reranked.append(new_result)

        return reranked

    @property
    def model_name(self) -> str:
        return self._model_name