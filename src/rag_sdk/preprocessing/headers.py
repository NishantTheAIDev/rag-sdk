"""Header/footer removal preprocessor."""

from __future__ import annotations

from difflib import SequenceMatcher

from rag_sdk.core import Document
from rag_sdk.preprocessing.base import Preprocessor


class HeaderFooterRemover(Preprocessor):
    """Detect and remove repeated headers/footers across pages/documents."""

    def __init__(self, similarity_threshold: float = 0.8, min_length: int = 20):
        self.similarity_threshold = similarity_threshold
        self.min_length = min_length

    def process(self, documents: list[Document]) -> list[Document]:
        if len(documents) < 2:
            return documents

        # Extract candidate headers/footers from each document
        candidates = self._find_repeated_segments(documents)
        
        if not candidates:
            return documents

        processed: list[Document] = []
        for doc in documents:
            text = doc.text
            for segment in candidates:
                # Remove the segment from the text
                text = text.replace(segment, "")
            # Clean up extra whitespace
            import re
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()
            
            processed.append(
                Document(
                    id=doc.id,
                    text=text,
                    metadata=doc.metadata,
                )
            )
        return processed

    def _find_repeated_segments(self, documents: list[Document]) -> list[str]:
        """Find text segments that appear in multiple documents (likely headers/footers)."""
        # Get first and last N lines from each document
        all_segments: list[str] = []
        
        for doc in documents:
            lines = doc.text.splitlines()
            # Check first 5 lines and last 5 lines
            for i in range(min(5, len(lines))):
                line = lines[i].strip()
                if len(line) >= self.min_length:
                    all_segments.append(line)
            for i in range(max(0, len(lines) - 5), len(lines)):
                line = lines[i].strip()
                if len(line) >= self.min_length:
                    all_segments.append(line)

        # Find segments that appear in multiple documents
        repeated: list[str] = []
        for segment in all_segments:
            count = sum(1 for doc in documents if segment in doc.text)
            if count >= max(2, len(documents) // 2):
                # Check if not already added (similarity check)
                is_duplicate = False
                for existing in repeated:
                    if self._similarity(segment, existing) >= self.similarity_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    repeated.append(segment)
        
        return repeated

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()