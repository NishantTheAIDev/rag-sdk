"""Multi-query retrieval wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_sdk.config import MultiQueryConfig
from rag_sdk.core import Chunk
from rag_sdk.retrieval.base import RetrievalResult, Retriever

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
        if top_k < 1:
            return []
        if not self._config.enabled:
            return self._base_retriever.search(query, top_k)

        queries = self._generate_queries(query)

        # Each sub-query must retrieve at least ``top_k`` results so the fused
        # ranking can still fill ``top_k`` slots.
        all_results = [self._base_retriever.search(q, top_k) for q in queries]
        return self._fuse_results(query, all_results, top_k)

    def _generate_queries(self, query: str) -> list[str]:
        """Generate up to ``num_queries`` distinct queries, original first."""
        queries = [query]
        if self._config.query_generator == "llm":
            if self._generator is None:
                raise ValueError("Multi-query with query_generator='llm' requires a generator")
            prompt = self._config.template or (
                "Generate {n} diverse search queries for the following question. "
                "Each query should be on a new line.\n\nQuestion: {query}\n\nQueries:"
            )
            formatted = prompt.format(n=self._config.num_queries - 1, query=query)
            response = self._generator.generate(formatted, [])
            queries.extend(line.strip() for line in response.text.splitlines())
        elif self._config.template:
            queries.append(self._config.template.format(query=query))

        unique: list[str] = []
        for candidate in queries:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique[: self._config.num_queries]

    def _fuse_results(
        self,
        query: str,
        all_results: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse N rankings using RRF or equal-weight min-max normalized scores."""
        if not all_results:
            return []
        if len(all_results) == 1:
            return all_results[0][:top_k]

        fused: dict[str, list] = {}
        for results in all_results:
            if self._config.fusion_method == "rrf":
                contributions = [
                    1.0 / (self._config.rrf_k + rank)
                    for rank in range(1, len(results) + 1)
                ]
            else:
                contributions = _min_max([r.score for r in results])
            for result, contribution in zip(results, contributions, strict=True):
                entry = fused.get(result.chunk.id)
                if entry is None:
                    fused[result.chunk.id] = [contribution, result]
                else:
                    entry[0] += contribution

        ranked = sorted(fused.values(), key=lambda entry: entry[0], reverse=True)
        return [
            result.model_copy(update={"score": score, "query": query})
            for score, result in ranked[:top_k]
        ]


def _min_max(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]
