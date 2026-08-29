"""Preprocessor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_sdk.core import Document


class Preprocessor(ABC):
    """Abstract base class for document preprocessors."""

    @abstractmethod
    def process(self, documents: list[Document]) -> list[Document]:
        """Process a list of documents, returning modified documents."""