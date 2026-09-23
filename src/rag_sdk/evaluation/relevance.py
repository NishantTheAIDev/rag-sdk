"""Pairing retrieved results with ground truth for retrieval metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from rag_sdk.dataset import QuerySample
from rag_sdk.retrieval import RetrievalResult

RelevanceLevel = Literal["document", "chunk"]


def judge_retrieval(
    retrieved: Sequence[RetrievalResult],
    sample: QuerySample,
    chunks_by_document: Mapping[str, Sequence[str]],
    level: RelevanceLevel = "document",
) -> tuple[list[str], set[str] | dict[str, int]]:
    """Return ``(ranked_ids, relevance)`` for one query at the given level.

    ``document``: retrieved chunks collapse to their documents (the first
    occurrence keeps its rank) and are scored against ``relevant_documents``,
    so a document split into many chunks is not over-counted.

    ``chunk``: chunk ids are scored against ``relevance_grades`` (graded) or
    ``relevant_chunks`` when the query provides them; otherwise every chunk of
    a relevant document counts as relevant, which favours larger chunks.
    """
    if level == "document":
        ranked_documents = list(dict.fromkeys(r.chunk.document_id for r in retrieved))
        return ranked_documents, set(sample.relevant_documents)

    retrieved_ids = [r.chunk.id for r in retrieved]
    if sample.relevance_grades:
        graded = dict(sample.relevance_grades)
        for chunk_id in sample.relevant_chunks:
            graded.setdefault(chunk_id, 1)
        return retrieved_ids, graded
    if sample.relevant_chunks:
        return retrieved_ids, set(sample.relevant_chunks)
    return retrieved_ids, {
        chunk_id
        for document_id in sample.relevant_documents
        for chunk_id in chunks_by_document.get(document_id, [])
    }
