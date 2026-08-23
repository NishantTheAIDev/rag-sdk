"""Ingestion pipeline for indexing documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_sdk.core import Chunk

if TYPE_CHECKING:
    from rag_sdk.chunking.base import Chunker
    from rag_sdk.chunking.parent_child import ParentChildChunks
    from rag_sdk.core import Document
    from rag_sdk.embeddings import EmbeddingProvider
    from rag_sdk.indexing import ChunkStore, DocumentStore, VectorStore


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    total_documents: int
    total_chunks: int
    total_parents: int
    total_children: int
    all_chunks: list[Chunk] = None  # All chunks for reference


def flatten_chunks(chunks: list[Chunk] | ParentChildChunks) -> list[Chunk]:
    """Flatten chunks from any chunker into a single list."""
    from rag_sdk.chunking.parent_child import ParentChildChunks as PCC
    if isinstance(chunks, PCC):
        return chunks.parents + chunks.children
    return list(chunks)


def ingest_documents(
    documents: list[Document],
    chunker: Chunker,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunk_store: ChunkStore | None = None,
    document_store: DocumentStore | None = None,
) -> IngestionResult:
    """Ingest documents: chunk, embed, index, and persist parents.

    For ParentChildChunker:
    - Children are embedded and added to VectorStore
    - Parents are persisted to ChunkStore only (not VectorStore)
    - All documents stored in DocumentStore for sentence-window expansion

    For other chunkers:
    - All chunks embedded and added to VectorStore
    - All documents stored in DocumentStore
    """
    from rag_sdk.chunking.parent_child import ParentChildChunks
    from rag_sdk.chunking.text import compute_sentence_boundaries

    total_chunks = 0
    total_parents = 0
    total_children = 0
    all_chunks: list[Chunk] = []

    for doc in documents:
        # Store full document for sentence-window expansion
        if document_store:
            document_store.add_document(doc)

        chunked = chunker.chunk(doc)

        # Compute sentence boundaries for ALL chunks
        flat_chunks = flatten_chunks(chunked)
        for chunk in flat_chunks:
            chunk.metadata.sentence_boundaries = compute_sentence_boundaries(
                doc.text, chunk.start_char, chunk.end_char
            )

        if isinstance(chunked, ParentChildChunks):
            # ParentChildChunker: embed and index children only
            child_texts = [c.text for c in chunked.children]
            child_vectors = embedding_provider.embed(child_texts)
            child_ids = [c.id for c in chunked.children]
            vector_store.add(child_ids, child_vectors)

            # Persist parents to ChunkStore only
            if chunk_store:
                chunk_store.add_parent_chunks(chunked.parents)

            total_parents += len(chunked.parents)
            total_children += len(chunked.children)
            total_chunks += len(chunked.parents) + len(chunked.children)
            all_chunks.extend(flat_chunks)
        else:
            # Standard chunking: embed and index all chunks
            texts = [c.text for c in flat_chunks]
            vectors = embedding_provider.embed(texts)
            ids = [c.id for c in flat_chunks]
            vector_store.add(ids, vectors)

            total_chunks += len(flat_chunks)
            all_chunks.extend(flat_chunks)

    return IngestionResult(
        total_documents=len(documents),
        total_chunks=total_chunks,
        total_parents=total_parents,
        total_children=total_children,
        all_chunks=all_chunks,
    )