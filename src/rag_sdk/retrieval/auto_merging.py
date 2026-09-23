"""Auto-merging retrieval post-processor."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_sdk.chunking.text import get_tokenizer
from rag_sdk.config import AutoMergingConfig
from rag_sdk.core import Chunk
from rag_sdk.indexing import VectorStore
from rag_sdk.retrieval.base import RetrievalResult


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def merge_chunks(a: Chunk, b: Chunk) -> Chunk:
    """Merge two adjacent chunks from the same document."""

    merged_text = a.text + " " + b.text
    return Chunk(
        id=f"{a.id}+{b.id}",
        text=merged_text,
        document_id=a.document_id,
        index=a.index,
        start_char=a.start_char,
        end_char=b.end_char,
        metadata=a.metadata.model_copy(),
    )


class AutoMerger:
    """Merges adjacent chunks from the same document based on embedding similarity."""

    def __init__(self, vector_store: VectorStore, config: AutoMergingConfig) -> None:
        self._vector_store = vector_store
        self._config = config
        self._tokenizer = get_tokenizer(config.tokenizer)

    def expand(self, results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        """Enricher entry point used by :class:`RetrievalPipeline`."""
        return self.merge(results)

    def merge(self, results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        # Group by document_id of SOURCE chunks (not current chunks after parent expansion)
        # The source chunk ID is always available and points to an indexed chunk
        by_doc: dict[str, list[RetrievalResult]] = {}
        for r in results:
            # We need the document_id of the source chunk
            # Since source_chunk_id is the ID of an indexed chunk, we can't directly
            # get its document_id without a lookup. For simplicity, use the current
            # chunk's document_id as proxy (it's the same document).
            # The key invariant: all source chunks for results from same document
            # will have the same document_id
            doc_id = r.chunk.document_id
            by_doc.setdefault(doc_id, []).append(r)

        merged_results: list[RetrievalResult] = []
        for _doc_id, doc_results in by_doc.items():
            # Sort by source chunk position (original retrieval order)
            doc_results.sort(key=lambda r: r.source_chunk_rank)

            i = 0
            while i < len(doc_results):
                current = doc_results[i]

                # Get embedding for SOURCE chunk (always in VectorStore)
                source_emb = self._vector_store.get_embedding(current.source_chunk_id)
                if source_emb is None:
                    merged_results.append(current)
                    i += 1
                    continue

                merged_chunk = current.chunk
                merged_source_ids = [current.source_chunk_id]
                merged_score = current.score
                merged_rerank = current.rerank_score

                j = i + 1
                token_limit = self._config.max_tokens
                while j < len(doc_results) and self._token_count(merged_chunk.text) < token_limit:
                    next_r = doc_results[j]
                    next_emb = self._vector_store.get_embedding(next_r.source_chunk_id)

                    if next_emb is not None:
                        sim = cosine_similarity(source_emb, next_emb)
                        if sim >= self._config.similarity_threshold:
                            merged_chunk = merge_chunks(merged_chunk, next_r.chunk)
                            merged_source_ids.append(next_r.source_chunk_id)
                            j += 1
                            continue
                    break

                merged_results.append(
                    RetrievalResult(
                        query=current.query,
                        chunk=merged_chunk,
                        score=merged_score,
                        source_chunk_id=current.source_chunk_id,
                        source_chunk_score=merged_score,
                        source_chunk_rank=current.source_chunk_rank,
                        rerank_score=merged_rerank,
                        rerank_rank=current.rerank_rank,
                        parent_id=current.parent_id,
                        child_ids=current.child_ids,
                        merged_source_ids=merged_source_ids,
                        expansion_type="auto_merged",
                    )
                )
                i = j

        return merged_results

    def _token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text))