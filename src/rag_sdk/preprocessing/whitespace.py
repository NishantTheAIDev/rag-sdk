"""Whitespace normalization preprocessor."""

from __future__ import annotations

import re

from rag_sdk.core import Document
from rag_sdk.preprocessing.base import Preprocessor


class WhitespaceNormalizer(Preprocessor):
    """Normalize whitespace in document text."""

    def process(self, documents: list[Document]) -> list[Document]:
        processed: list[Document] = []
        for doc in documents:
            # Replace multiple spaces with single space
            text = re.sub(r"[ \t]+", " ", doc.text)
            # Replace multiple newlines with double newline
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Strip leading/trailing whitespace from each line
            text = "\n".join(line.strip() for line in text.splitlines())
            # Remove leading/trailing whitespace from entire text
            text = text.strip()
            
            processed.append(
                Document(
                    id=doc.id,
                    text=text,
                    metadata=doc.metadata,
                )
            )
        return processed