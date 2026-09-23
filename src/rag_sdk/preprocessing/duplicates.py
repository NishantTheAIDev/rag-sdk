"""Duplicate detection preprocessor."""

from __future__ import annotations

from difflib import SequenceMatcher
from hashlib import sha256

from rag_sdk.core import Document
from rag_sdk.preprocessing.base import Preprocessor


class DuplicateDetector(Preprocessor):
    """Detect and remove duplicate documents."""

    def __init__(self, similarity_threshold: float = 0.95, use_hash: bool = True):
        self.similarity_threshold = similarity_threshold
        self.use_hash = use_hash

    def process(self, documents: list[Document]) -> list[Document]:
        if len(documents) < 2:
            return documents

        seen_hashes: set[str] = set()
        unique_docs: list[Document] = []

        for doc in documents:
            if self.use_hash:
                doc_hash = self._hash_document(doc)
                if doc_hash in seen_hashes:
                    continue  # exact duplicate
                seen_hashes.add(doc_hash)
            # Near-duplicate check; a threshold of 1.0 means exact matches only.
            if self.similarity_threshold < 1.0 and any(
                self._similarity(doc.text, existing.text) >= self.similarity_threshold
                for existing in unique_docs
            ):
                continue
            unique_docs.append(doc)

        return unique_docs

    def _hash_document(self, doc: Document) -> str:
        """Generate a hash of the document text."""
        return sha256(doc.text.encode("utf-8")).hexdigest()

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()