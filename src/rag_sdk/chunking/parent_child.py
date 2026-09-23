"""Parent-child chunking strategy."""

from __future__ import annotations

from dataclasses import dataclass

from rag_sdk.chunking.base import Chunker, build_chunks
from rag_sdk.chunking.recursive import RecursiveChunker
from rag_sdk.chunking.text import apply_overlap, recursive_split
from rag_sdk.config import (
    ParentChildChunkerConfig,
    RecursiveChunkerConfig,
    SentenceWindowChunkerConfig,
)
from rag_sdk.core import Chunk, Document


@dataclass
class ParentChildChunks:
    """Result of parent-child chunking."""

    parents: list[Chunk]
    children: list[Chunk]


class ParentChildChunker(Chunker):
    """Produces parent chunks and child chunks with explicit relationships.

    Parent chunks are larger and persisted to ChunkStore for retrieval-time expansion.
    Child chunks are smaller and indexed in the VectorStore for retrieval.
    """

    def __init__(
        self,
        parent_chunk_size: int,
        parent_overlap: int,
        child_chunk_size: int,
        child_overlap: int,
        separators: list[str] | None = None,
    ) -> None:
        self._parent_chunker = RecursiveChunker(
            RecursiveChunkerConfig(
                strategy="recursive",
                chunk_size=parent_chunk_size,
                overlap=parent_overlap,
                separators=separators or ["\n\n", "\n", ". ", " "],
            )
        )
        self._child_chunk_size = child_chunk_size
        self._child_overlap = child_overlap
        self._separators = separators or ["\n\n", "\n", ". ", " "]

    @classmethod
    def from_config(cls, config: ParentChildChunkerConfig) -> ParentChildChunker:
        return cls(
            parent_chunk_size=config.parent_chunk_size,
            parent_overlap=config.parent_overlap,
            child_chunk_size=config.child_chunk_size,
            child_overlap=config.child_overlap,
        )

    def chunk(self, document: Document) -> ParentChildChunks:
        # First, create parent chunks
        parent_pieces = recursive_split(
            document.text, self._separators, self._parent_chunker._chunk_size
        )
        parent_pieces = apply_overlap(parent_pieces, self._parent_chunker._overlap)
        parent_chunks = build_chunks(document, parent_pieces)

        # Mark parent chunks
        for _i, chunk in enumerate(parent_chunks):
            chunk.metadata.chunk_type = "parent"
            chunk.metadata.child_ids = []

        # For each parent, create child chunks
        child_chunks: list[Chunk] = []
        for parent_chunk in parent_chunks:
            # Get the text for this parent
            parent_text = parent_chunk.text
            parent_start = parent_chunk.start_char

            # Split parent into children
            child_pieces = recursive_split(parent_text, self._separators, self._child_chunk_size)
            child_pieces = apply_overlap(child_pieces, self._child_overlap)

            # Build child chunks with document-relative offsets
            for idx, (text, start, end) in enumerate(child_pieces):
                child_id = f"{document.id}:p{parent_chunk.index}:c{idx}"
                child_chunk = Chunk(
                    id=child_id,
                    text=text,
                    document_id=document.id,
                    index=len(child_chunks),
                    start_char=parent_start + start,
                    end_char=parent_start + end,
                    metadata=document.metadata.model_copy(),
                )
                child_chunk.metadata.chunk_type = "child"
                child_chunk.metadata.parent_id = parent_chunk.id
                child_chunks.append(child_chunk)

            # Update parent's child_ids
            parent_chunk.metadata.child_ids = [
                c.id for c in child_chunks if c.metadata.parent_id == parent_chunk.id
            ]

        return ParentChildChunks(parents=parent_chunks, children=child_chunks)


class SentenceWindowChunker(Chunker):
    """Produces overlapping sentence windows as chunks."""

    def __init__(
        self,
        window_size: int,
        window_overlap: int,
    ) -> None:
        self._window_size = window_size
        self._window_overlap = window_overlap

    @classmethod
    def from_config(cls, config: SentenceWindowChunkerConfig) -> SentenceWindowChunker:
        return cls(
            window_size=config.window_size,
            window_overlap=config.window_overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        from rag_sdk.chunking.text import split_sentences

        sentences = split_sentences(document.text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        i = 0
        while i < len(sentences):
            # Build window of sentences
            window_sentences = sentences[i : i + self._window_size]
            if not window_sentences:
                break

            window_text = " ".join(s[0] for s in window_sentences)
            start = window_sentences[0][1]
            end = window_sentences[-1][2]

            chunk = Chunk(
                id=f"{document.id}:sw{i}",
                text=window_text,
                document_id=document.id,
                index=len(chunks),
                start_char=start,
                end_char=end,
                metadata=document.metadata.model_copy(),
            )
            chunk.metadata.chunk_type = "sentence_window"
            chunks.append(chunk)

            # Move by window_size - overlap
            i += self._window_size - self._window_overlap
            if i >= len(sentences):
                break

        return chunks