"""Text cleanup preprocessor."""

from __future__ import annotations

import re

from rag_sdk.core import Document
from rag_sdk.preprocessing.base import Preprocessor


class TextCleanup(Preprocessor):
    """Clean up text: remove control characters, fix encoding issues."""

    def process(self, documents: list[Document]) -> list[Document]:
        processed: list[Document] = []
        for doc in documents:
            text = doc.text
            
            # Remove control characters except newlines and tabs
            text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
            
            # Fix common encoding issues
            text = text.replace("\ufeff", "")  # BOM
            text = text.replace("\u200b", "")  # Zero-width space
            
            # Normalize unicode
            import unicodedata
            text = unicodedata.normalize("NFC", text)
            
            processed.append(
                Document(
                    id=doc.id,
                    text=text,
                    metadata=doc.metadata,
                )
            )
        return processed