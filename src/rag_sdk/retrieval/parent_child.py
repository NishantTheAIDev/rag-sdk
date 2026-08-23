"""Parent-child retrieval expander."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.core import Chunk
from rag_sdk.indexing import ChunkStore
from rag_sdk.retrieval.base import RetrievalResult


class ParentChildExpander:
    """Expands child chunk results to their parent chunks."""

    def __init__(self, chunk_store: ChunkStore) -> None:
        self._chunk_store = chunk_store

    def expand(self, results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        # First enrichment - source chunks are in r.chunk
        return self.expand_with_sources(results, {})

    def expand_with_sources(
        self,
        results: Sequence[RetrievalResult],
        source_chunk_map: dict[str, Chunk],
    ) -> list[RetrievalResult]:
        # Map source_chunk_id -> parent_id using chunk metadata
        parent_ids: list[str] = []
        source_to_parent: dict[str, list[RetrievalResult]] = {}

        for r in results:
            # Source chunk is the ORIGINAL retrieved chunk
            # At this stage (first enrichment), r.chunk is the source chunk
            source_chunk = r.chunk
            if source_chunk.metadata.chunk_type == "child":
                pid = source_chunk.metadata.parent_id
                if pid and pid not in parent_ids:
                    parent_ids.append(pid)
                if pid:
                    source_to_parent.setdefault(pid, []).append(r)

        if not parent_ids:
            return list(results)

        parent_chunks = self._chunk_store.get_parent_chunks(parent_ids)
        parent_map = {p.id: p for p in parent_chunks}

        parent_results = []
        for pid in parent_ids:
            parent_chunk = parent_map.get(pid)
            if not parent_chunk:
                continue

            child_results = source_to_parent.get(pid, [])
            # Aggregate scores from source chunks
            best_score = max(r.source_chunk_score for r in child_results) if child_results else 0.0
            best_rerank = max(
                (r.rerank_score for r in child_results if r.rerank_score is not None), default=None
            )
            best_orig_rank = min(r.source_chunk_rank for r in child_results) if child_results else 0
            best_rerank_rank = min(
                (r.rerank_rank for r in child_results if r.rerank_rank is not None), default=None
            )
            source_ids = [r.source_chunk_id for r in child_results]

            parent_results.append(
                RetrievalResult(
                    query=results[0].query if results else "",
                    chunk=parent_chunk,
                    score=best_score,
                    source_chunk_id=source_ids[0] if source_ids else "",
                    source_chunk_score=best_score,
                    source_chunk_rank=best_orig_rank,
                    rerank_score=best_rerank,
                    rerank_rank=best_rerank_rank,
                    parent_id=pid,
                    child_ids=source_ids,
                    expansion_type="parent_child",
                )
            )

        return parent_results