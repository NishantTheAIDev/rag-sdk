"""Multi-query retrieval wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_sdk.config import MultiQueryConfig
from rag_sdk.core import Chunk
from rag_sdk.retrieval.base import RetrievalResult, Retriever
from rag_sdk.retrieval.fusion import rrf_fuse, weighted_fuse

if TYPE_CHECKING:
    from rag_sdk.generation import Generator
    from rag_sdk.retrieval.base import Retriever as BaseRetriever


class MultiQueryRetriever(Retriever):
    """Wraps a base retriever to generate and execute multiple queries.

    Generates multiple queries from the original query, retrieves for each,
    then fuses the results using RRF or weighted fusion.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        config: MultiQueryConfig,
        generator: Generator | None = None,
    ) -> None:
        self._base_retriever = base_retriever
        self._config = config
        self._generator = generator

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._base_retriever.add_chunks(chunks)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self._config.enabled:
            return self._base_retriever.search(query, top_k)

        # Generate multiple queries
        queries = self._generate_queries(query)
        
        # Retrieve for each query
        all_results: list[list[RetrievalResult]] = []
        for q in queries:
            results = self._base_retriever.search(q, self._config.num_queries)
            all_results.append(results)

        # Fuse results
        fused = self._fuse_results(all_results, top_k)
        return fused

    def _generate_queries(self, query: str) -> list[str]:
        """Generate multiple queries from the original."""
        if self._config.query_generator == "llm" and self._generator:
            prompt = self._config.template or (
                "Generate {n} diverse search queries for the following question. "
                "Each query should be on a new line.\n\nQuestion: {query}\n\nQueries:"
            )
            formatted = prompt.format(n=self._config.num_queries, query=query)
            response = self._generator.generate(formatted, [])
            generated = [q.strip() for q in response.text.splitlines() if q.strip()]
            return [query] + generated[: self._config.num_queries - 1]
        
        elif self._config.query_generator == "template" and self._config.template:
            # Simple template-based generation
            queries = [
                self._config.template.format(query=query)
                for _ in range(self._config.num_queries)
            ]
            return [query] + queries
        
        # Fallback: return just the original query
        return [query]

    def _fuse_results(
        self,
        all_results: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse results from multiple queries using RRF or weighted fusion."""
        if not all_results:
            return []
        if len(all_results) == 1:
            return all_results[0][:top_k]

        # Convert to format expected by fusion functions
        # each result list -> list of (chunk_id, score)
        result_lists = [
            [(r.chunk.id, r.score) for r in results]
            for results in all_results
        ]

        if self._config.fusion_method == "rrf":
            fused = rrf_fuse(result_lists, k=self._config.get("rrf_k", 60))
        else:
            weights = self._config.get("weights", [1.0] * len(result_lists))
            fused = weighted_fuse(result_lists, weights)

        # Map back to RetrievalResult objects
        # Build a lookup of chunk_id -> RetrievalResult from the first query
        chunk_lookup = {r.chunk.id: r for r in all_results[0]}
        for results in all_results[1:]:
            for r in results:
                if r.chunk.id not in chunk_lookup:
                    chunk_lookup[r.chunk.id] = r

        final_results = []
        for chunk_id, score in fused[:top_k]:
            if chunk_id in chunk_lookup:
                result = chunk_lookup[chunk_id]
                result.score = score  # Update with fused score
                final_results.append(result)

        return final_results