"""Semantic chunking using embedding similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rag_sdk.chunking.base import Chunker
from rag_sdk.config import SemanticChunkerConfig
from rag_sdk.core import Chunk, Document

if TYPE_CHECKING:
    from rag_sdk.embeddings import EmbeddingProvider


class SemanticChunker(Chunker):
    """Chunk documents based on semantic similarity between sentences.

    Algorithm:
    1. Split document into sentences
    2. Embed each sentence (or window of sentences)
    3. Compute cosine similarity between adjacent sentence embeddings
    4. Group sentences into chunks where similarity > threshold
    5. Respect min/max chunk size constraints
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        similarity_threshold: float,
        min_chunk_size: int,
        max_chunk_size: int,
        embedding_provider: EmbeddingProvider,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.embedding_provider = embedding_provider

    @classmethod
    def from_config(cls, config: SemanticChunkerConfig) -> SemanticChunker:
        """Build a semantic chunker from configuration.

        Uses the global embedding provider from the config.
        """
        # Import here to avoid circular imports
        from rag_sdk.config import EmbeddingConfig
        from rag_sdk.embeddings import build_embedding_provider

        embedding_provider = build_embedding_provider(EmbeddingConfig(
            provider=config.embedding_provider,
            model=config.embedding_model,
        ))

        return cls(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            similarity_threshold=config.similarity_threshold,
            min_chunk_size=config.min_chunk_size,
            max_chunk_size=config.max_chunk_size,
            embedding_provider=embedding_provider,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into semantically coherent chunks."""
        # Step 1: Split into sentences
        sentences = self._split_sentences(document.text)
        if not sentences:
            return []

        # Step 2: Embed sentences
        sentence_texts = [s[0] for s in sentences]
        embeddings = self.embedding_provider.embed(sentence_texts)

        # Step 3: Compute boundaries based on similarity
        boundaries = self._compute_boundaries(embeddings)

        # Step 4: Build chunks from sentence groups
        chunks = self._build_chunks_from_boundaries(document, sentences, boundaries)

        # Step 5: Apply overlap and size constraints
        chunks = self._apply_constraints(chunks, document)

        return chunks

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with character offsets.
        
        Returns list of (sentence_text, start_char, end_char).
        """
        # Reuse the existing sentence splitter from text.py
        from rag_sdk.chunking.text import split_sentences
        return split_sentences(text)

    def _compute_boundaries(self, embeddings: np.ndarray) -> list[int]:
        """Compute chunk boundaries based on embedding similarity.
        
        Returns list of indices where new chunks should start.
        """
        if len(embeddings) <= 1:
            return [0]

        boundaries = [0]  # First sentence always starts a chunk

        for i in range(1, len(embeddings)):
            # Cosine similarity between adjacent sentence embeddings
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                boundaries.append(i)

        return boundaries

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _build_chunks_from_boundaries(
        self,
        document: Document,
        sentences: list[tuple[str, int, int]],
        boundaries: list[int],
    ) -> list[Chunk]:
        """Build chunks from sentence groups defined by boundaries."""
        chunks = []
        for idx, start_idx in enumerate(boundaries):
            end_idx = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(sentences)
            
            # Combine sentences in this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(s[0] for s in chunk_sentences)
            start_char = chunk_sentences[0][1]
            end_char = chunk_sentences[-1][2]
            
            chunks.append(Chunk(
                id=f"{document.id}:{len(chunks)}",
                text=chunk_text,
                document_id=document.id,
                index=len(chunks),
                start_char=start_char,
                end_char=end_char,
                metadata=document.metadata,
            ))
        
        return chunks

    def _apply_constraints(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """Apply min/max chunk size constraints."""
        if not chunks:
            return chunks
        
        # First pass: handle chunks that are too large
        temp_chunks = []
        for chunk in chunks:
            if len(chunk.text) <= self.max_chunk_size:
                temp_chunks.append(chunk)
            else:
                # Split large chunk using recursive chunking as fallback
                from rag_sdk.chunking.recursive import RecursiveChunker
                from rag_sdk.config import RecursiveChunkerConfig
                fallback_config = RecursiveChunkerConfig(
                    strategy="recursive",
                    chunk_size=self.max_chunk_size,
                    overlap=self.overlap,
                )
                fallback = RecursiveChunker(fallback_config)
                sub_chunks = fallback.chunk(document)
                # Filter to only those within this chunk's span
                for sub in sub_chunks:
                    if sub.start_char >= chunk.start_char and sub.end_char <= chunk.end_char:
                        temp_chunks.append(sub)
        
        # Second pass: merge chunks that are too small
        final_chunks = []
        i = 0
        while i < len(temp_chunks):
            chunk = temp_chunks[i]
            if len(chunk.text) >= self.min_chunk_size:
                final_chunks.append(chunk)
                i += 1
            else:
                # Try to merge with next chunk
                if i + 1 < len(temp_chunks):
                    next_chunk = temp_chunks[i + 1]
                    merged_text = chunk.text + " " + next_chunk.text
                    if len(merged_text) <= self.max_chunk_size:
                        # Merge them
                        merged_chunk = Chunk(
                            id=f"{document.id}:{len(final_chunks)}",
                            text=merged_text,
                            document_id=document.id,
                            index=len(final_chunks),
                            start_char=chunk.start_char,
                            end_char=next_chunk.end_char,
                            metadata=document.metadata,
                        )
                        final_chunks.append(merged_chunk)
                        i += 2
                    else:
                        # Can't merge, keep as is even if small
                        final_chunks.append(chunk)
                        i += 1
                else:
                    # Last chunk and it's small, keep as is
                    final_chunks.append(chunk)
                    i += 1
        
        # Re-index
        for i, chunk in enumerate(final_chunks):
            chunk.index = i
            chunk.id = f"{document.id}:{i}"
        
        return final_chunks