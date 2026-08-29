"""Metadata extraction preprocessor."""

from __future__ import annotations

import re

from rag_sdk.core import Document
from rag_sdk.preprocessing.base import Preprocessor


class MetadataExtractor(Preprocessor):
    """Extract metadata from document text patterns."""

    def process(self, documents: list[Document]) -> list[Document]:
        processed: list[Document] = []
        for doc in documents:
            metadata = doc.metadata
            text = doc.text
            
            # Extract email addresses
            emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
            if emails:
                metadata = metadata.model_copy(update={"emails": list(set(emails))})
            
            # Extract URLs
            urls = re.findall(r"https?://[^\s]+", text)
            if urls:
                metadata = metadata.model_copy(update={"urls": list(set(urls))})
            
            # Extract dates (simple patterns)
            dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b", text)
            if dates:
                metadata = metadata.model_copy(update={"dates": list(set(dates))})
            
            # Extract potential titles (first line if it looks like a title)
            first_line = text.splitlines()[0].strip() if text.splitlines() else ""
            is_title = (
                first_line
                and len(first_line) < 100
                and not first_line.endswith(".")
                and not metadata.title
            )
            if is_title:
                metadata = metadata.model_copy(update={"title": first_line})
            
            processed.append(
                Document(
                    id=doc.id,
                    text=text,
                    metadata=metadata,
                )
            )
        return processed