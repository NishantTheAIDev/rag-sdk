"""Maximal Marginal Relevance (MMR) retriever."""

from __future__ import annotations

import faiss
import numpy as np

from rag_sdk.config import MMRRetrievalConfig
from rag_sdk.core import Chunk, matches_filters
from rag_sdk.indexing import VectorStore
from rag_sdk.retrieval.base import RetrievalResult, Retriever


class MMRRetriever(Retriever):
    """Retriever using Maximal Marginal Relevance for diversity.

    MMR balances relevance to query with diversity among results.
    lambda_param = 1.0 -> pure relevance (like dense retrieval)
    lambda_param = 0.0 -> pure diversity
    """

    def __init__(
        self,
        config: MMRRetrievalConfig,
        embedding_provider,
        vector_store: VectorStore,
    ) -> None:
        self._config = config
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunks: list[Chunk] = []
        self._chunk_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Register chunks and compute their embeddings."""
        self._chunks = list(chunks)
        self._chunk_ids = [c.id for c in chunks]
        if chunks:
            embeddings = []
            for chunk in chunks:
                emb = self._vector_store.get_embedding(chunk.id)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    emb = self._embedding_provider.embed([chunk.text])[0]
                    embeddings.append(emb)
            self._embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(self._embeddings)
        else:
            self._embeddings = None

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Search using MMR.

        1. Get candidate_k results from dense retrieval
        2. Apply MMR to select top_k diverse results
        """
        if not self._chunks or self._embeddings is None:
            return []

        candidate_k = min(self._config.candidate_k, len(self._chunks))
        query_embedding = self._embedding_provider.embed([query])[0]
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        scores = self._embeddings @ query_embedding.T
        scores = scores.flatten()

        ranked = np.argsort(scores)[::-1]
        if self._config.filters:
            ranked = np.array(
                [
                    idx
                    for idx in ranked
                    if matches_filters(self._chunks[idx].metadata, self._config.filters)
                ],
                dtype=np.int64,
            )
            if ranked.size == 0:
                return []
        candidate_indices = ranked[:candidate_k]
        candidate_scores = scores[candidate_indices]

        selected = self._mmr_select(
            query_embedding=query_embedding.flatten(),
            candidate_embeddings=self._embeddings[candidate_indices],
            candidate_scores=candidate_scores,
            top_k=top_k,
            lambda_param=self._config.lambda_param,
        )

        results = []
        for rank, idx in enumerate(selected):
            chunk_idx = candidate_indices[idx]
            chunk = self._chunks[chunk_idx]
            score = float(candidate_scores[idx])
            results.append(
                RetrievalResult(
                    query=query,
                    chunk=chunk,
                    score=score,
                    source_chunk_id=chunk.id,
                    source_chunk_score=score,
                    source_chunk_rank=rank,
                )
            )
        return results

    def _mmr_select(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_scores: np.ndarray,
        top_k: int,
        lambda_param: float,
    ) -> list[int]:
        """Select diverse results using MMR.

        Returns indices into candidate_embeddings.
        """
        n_candidates = candidate_embeddings.shape[0]
        if top_k >= n_candidates:
            return list(range(n_candidates))

        selected = []
        remaining = list(range(n_candidates))

        first_idx = int(np.argmax(candidate_scores))
        selected.append(first_idx)
        remaining.remove(first_idx)

        while len(selected) < top_k and remaining:
            best_score = -np.inf
            best_idx = -1

            for idx in remaining:
                relevance = candidate_scores[idx]
                max_similarity = 0.0
                if selected:
                    similarities = candidate_embeddings[idx] @ candidate_embeddings[selected].T
                    max_similarity = float(np.max(similarities))
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected


try:
    import faiss
except ImportError:
    faiss = None  # type: ignore