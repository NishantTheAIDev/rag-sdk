"""Cohere reranker adapter."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.config import CohereRerankerConfig
from rag_sdk.reranking.base import RerankerProvider
from rag_sdk.retrieval.base import RetrievalResult


class CohereReranker(RerankerProvider):
    """Reranker using Cohere's Rerank API."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere is required for CohereReranker. "
                "Install with 'pip install cohere'."
            ) from e

        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model
        self._model_name = model

    @classmethod
    def from_config(cls, config: CohereRerankerConfig) -> CohereReranker:
        return cls(model=config.model, api_key=config.api_key)

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        documents = [c.chunk.text for c in candidates]

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=False,
        )

        # Map results back to candidates
        reranked = []
        for result in response.results:
            idx = result.index
            candidate = candidates[idx]
            new_result = candidate.model_copy(deep=True)
            new_result.rerank_score = result.relevance_score
            new_result.rerank_rank = len(reranked) + 1
            reranked.append(new_result)

        return reranked

    @property
    def model_name(self) -> str:
        return self._model_name