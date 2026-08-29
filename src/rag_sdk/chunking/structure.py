"""Structure-aware chunking using document heading hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_sdk.chunking.base import Chunker
from rag_sdk.config import StructureAwareChunkerConfig
from rag_sdk.core import Chunk, Document

if TYPE_CHECKING:
    pass


class StructureAwareChunker(Chunker):
    """Chunk documents respecting heading hierarchy from document metadata.

    Algorithm:
    1. Use headings from document metadata (extracted during ingestion)
    2. Split document at heading boundaries
    3. Preserve heading context in each chunk (breadcrumb trail)
    4. Respect chunk_size/overlap within sections
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        include_heading_context: bool,
        max_heading_depth: int,
        split_on_headings: list[str],
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.include_heading_context = include_heading_context
        self.max_heading_depth = max_heading_depth
        self.split_on_headings = split_on_headings

    @classmethod
    def from_config(cls, config: StructureAwareChunkerConfig) -> StructureAwareChunker:
        return cls(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            include_heading_context=config.include_heading_context,
            max_heading_depth=config.max_heading_depth,
            split_on_headings=config.split_on_headings,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document at heading boundaries, preserving hierarchy."""
        # Get headings from metadata
        headings = document.metadata.headings
        
        if not headings:
            # No headings found, fall back to recursive chunking
            from rag_sdk.chunking.recursive import RecursiveChunker
            fallback = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.overlap)
            return fallback.chunk(document)

        # Find heading positions in text
        heading_positions = self._find_heading_positions(document.text, headings)
        
        if not heading_positions:
            # Could not locate headings, fall back
            from rag_sdk.chunking.recursive import RecursiveChunker
            fallback = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.overlap)
            return fallback.chunk(document)

        # Build sections from headings
        sections = self._build_sections(document, heading_positions)
        
        # Chunk each section
        all_chunks = []
        for section in sections:
            section_chunks = self._chunk_section(document, section)
            all_chunks.extend(section_chunks)
        
        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.index = i
            chunk.id = f"{document.id}:{i}"
        
        return all_chunks

    def _find_heading_positions(
        self, text: str, headings: list[str]
    ) -> list[tuple[str, int, int]]:
        """Find positions of headings in the document text."""
        positions = []
        for heading in headings:
            # Try exact match first
            pos = text.find(heading)
            if pos >= 0:
                positions.append((heading, pos, pos + len(heading)))
                continue
            
            # Try with markdown prefix
            for prefix in ["# ", "## ", "### ", "#### ", "##### ", "###### "]:
                marked = prefix + heading
                pos = text.find(marked)
                if pos >= 0:
                    positions.append((heading, pos, pos + len(marked)))
                    break
        
        # Sort by position
        positions.sort(key=lambda x: x[1])
        return positions

    def _build_sections(
        self,
        document: Document,
        heading_positions: list[tuple[str, int, int]],
    ) -> list[dict]:
        """Build section dicts from heading positions."""
        sections = []
        for i, (heading, _start, end) in enumerate(heading_positions):
            next_start = (
                heading_positions[i + 1][1]
                if i + 1 < len(heading_positions)
                else len(document.text)
            )
            section_text = document.text[end:next_start].strip()
            
            # Build heading context (breadcrumb)
            heading_context = heading
            if self.include_heading_context and i > 0:
                # Include parent headings up to max_heading_depth
                context_parts = [h for h, _, _ in heading_positions[:i+1]][-self.max_heading_depth:]
                heading_context = " > ".join(context_parts)
            
            sections.append({
                "heading": heading,
                "heading_context": heading_context,
                "text": section_text,
                "start_char": end,
                "end_char": next_start,
            })
        
        # Handle content before first heading
        if heading_positions and heading_positions[0][1] > 0:
            first_text = document.text[:heading_positions[0][1]].strip()
            if first_text:
                sections.insert(0, {
                    "heading": "",
                    "heading_context": "",
                    "text": first_text,
                    "start_char": 0,
                    "end_char": heading_positions[0][1],
                })
        
        return sections

    def _chunk_section(
        self,
        document: Document,
        section: dict,
    ) -> list[Chunk]:
        """Chunk a single section, respecting size limits."""
        text = section["text"]
        if not text:
            return []
        
        # Split section into chunks of appropriate size
        # Use sentence-aware splitting
        from rag_sdk.chunking.text import split_sentences
        sentences = split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        
        for sentence_text, sent_start, sent_end in sentences:
            sent_len = len(sentence_text)
            
            if current_length + sent_len > self.chunk_size and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(s[0] for s in current_chunk_sentences)
                chunk_start = section["start_char"] + current_chunk_sentences[0][1]
                chunk_end = section["start_char"] + current_chunk_sentences[-1][2]
                
                # Build metadata with heading context
                metadata = document.metadata.model_copy()
                if self.include_heading_context and section["heading_context"]:
                    metadata = metadata.model_copy(update={
                        "section_hierarchy": section["heading_context"]
                    })
                
                chunks.append(Chunk(
                    id=f"{document.id}:{len(chunks)}",  # Will be re-indexed
                    text=chunk_text,
                    document_id=document.id,
                    index=len(chunks),
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata=metadata,
                ))
                
                # Handle overlap
                if self.overlap > 0 and len(current_chunk_sentences) > 1:
                    # Keep last few sentences for overlap
                    overlap_text = " ".join(s[0] for s in current_chunk_sentences[-2:])
                    if len(overlap_text) <= self.overlap:
                        current_chunk_sentences = current_chunk_sentences[-2:]
                        current_length = len(overlap_text)
                    else:
                        current_chunk_sentences = [current_chunk_sentences[-1]]
                        current_length = len(current_chunk_sentences[0][0])
                else:
                    current_chunk_sentences = []
                    current_length = 0
            
            current_chunk_sentences.append((sentence_text, sent_start, sent_end))
            current_length += sent_len
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(s[0] for s in current_chunk_sentences)
            chunk_start = section["start_char"] + current_chunk_sentences[0][1]
            chunk_end = section["start_char"] + current_chunk_sentences[-1][2]
            
            metadata = document.metadata.model_copy()
            if self.include_heading_context and section["heading_context"]:
                metadata = metadata.model_copy(update={
                    "section_hierarchy": section["heading_context"]
                })
            
            chunks.append(Chunk(
                id=f"{document.id}:{len(chunks)}",
                text=chunk_text,
                document_id=document.id,
                index=len(chunks),
                start_char=chunk_start,
                end_char=chunk_end,
                metadata=metadata,
            ))
        
        return chunks