"""Sentence window retrieval expander."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.config import SentenceWindowExpansionConfig
from rag_sdk.core import Chunk
from rag_sdk.indexing import DocumentStore
from rag_sdk.retrieval.base import RetrievalResult


class SentenceWindowExpander:
    """Expands chunks to include surrounding sentences using stored boundaries."""

    def __init__(
        self, document_store: DocumentStore, config: SentenceWindowExpansionConfig
    ) -> None:
        self._document_store = document_store
        self._config = config

    def expand(self, results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        # Fallback for direct calls without source map
        return self.expand_with_sources(results, {})

    def expand_with_sources(
        self,
        results: Sequence[RetrievalResult],
        source_chunk_map: dict[str, Chunk],
    ) -> list[RetrievalResult]:
        expanded = []
        for r in results:
            # Get the original source chunk from the map
            source_chunk = source_chunk_map.get(r.source_chunk_id)
            if not source_chunk:
                # Fallback to current chunk
                source_chunk = r.chunk

            boundaries = source_chunk.metadata.sentence_boundaries
            if not boundaries:
                expanded.append(r)
                continue

            doc = self._document_store.get_document(source_chunk.document_id)
            if not doc:
                expanded.append(r)
                continue

            expanded_chunk = self._expand_to_window(
                source_chunk, doc.text, boundaries
            )

            expanded.append(
                RetrievalResult(
                    query=r.query,
                    chunk=expanded_chunk,
                    score=r.score,
                    source_chunk_id=r.source_chunk_id,
                    source_chunk_score=r.source_chunk_score,
                    source_chunk_rank=r.source_chunk_rank,
                    rerank_score=r.rerank_score,
                    rerank_rank=r.rerank_rank,
                    parent_id=r.parent_id,
                    child_ids=r.child_ids,
                    merged_source_ids=r.merged_source_ids,
                    expansion_type="sentence_window",
                )
            )
        return expanded

    def _expand_to_window(
        self, source_chunk: Chunk, document_text: str, boundaries: list[tuple[int, int]]
    ) -> Chunk:
        """Expand chunk to include surrounding sentences up to window_size."""

        if not boundaries:
            return source_chunk

        # Find the sentence index containing the chunk's start
        chunk_start = source_chunk.start_char
        chunk_end = source_chunk.end_char

        # Find first sentence that overlaps with chunk
        first_idx = 0
        for i, (_s_start, s_end) in enumerate(boundaries):
            if s_end > chunk_start:
                first_idx = i
                break

        # Find last sentence that overlaps with chunk
        last_idx = len(boundaries) - 1
        for i, (s_start, _s_end) in enumerate(boundaries):
            if s_start >= chunk_end:
                last_idx = i - 1
                break

        # Expand window around the chunk's sentences
        window_radius = (self._config.window_size - 1) // 2
        start_idx = max(0, first_idx - window_radius)
        end_idx = min(len(boundaries) - 1, last_idx + window_radius)

        # Ensure we have window_size sentences total
        while end_idx - start_idx + 1 < self._config.window_size:
            if start_idx > 0:
                start_idx -= 1
            elif end_idx < len(boundaries) - 1:
                end_idx += 1
            else:
                break

        # Extract expanded text
        expanded_start = boundaries[start_idx][0]
        expanded_end = boundaries[end_idx][1]
        expanded_text = document_text[expanded_start:expanded_end]

        return Chunk(
            id=f"{source_chunk.id}:sw",
            text=expanded_text,
            document_id=source_chunk.document_id,
            index=source_chunk.index,
            start_char=expanded_start,
            end_char=expanded_end,
            metadata=source_chunk.metadata.model_copy(),
        )