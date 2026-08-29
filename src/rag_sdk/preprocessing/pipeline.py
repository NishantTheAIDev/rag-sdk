"""Preprocessing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_sdk.core import Document
from rag_sdk.preprocessing.base import Preprocessor
from rag_sdk.preprocessing.cleanup import TextCleanup
from rag_sdk.preprocessing.duplicates import DuplicateDetector
from rag_sdk.preprocessing.headers import HeaderFooterRemover
from rag_sdk.preprocessing.metadata import MetadataExtractor
from rag_sdk.preprocessing.whitespace import WhitespaceNormalizer

if TYPE_CHECKING:
    from rag_sdk.config import PreprocessingConfig


class PreprocessingPipeline:
    """Fixed-order preprocessing pipeline.

    Order: normalize_whitespace -> cleanup_text -> remove_headers
    -> remove_duplicates -> extract_metadata
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._steps: list[Preprocessor] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the pipeline steps based on config."""
        if self.config.normalize_whitespace:
            self._steps.append(WhitespaceNormalizer())
        
        if self.config.cleanup_text:
            self._steps.append(TextCleanup())
        
        if self.config.remove_headers:
            self._steps.append(HeaderFooterRemover(
                similarity_threshold=self.config.header_footer_similarity
            ))
        
        if self.config.remove_duplicates:
            self._steps.append(DuplicateDetector(
                similarity_threshold=self.config.duplicate_similarity
            ))
        
        if self.config.extract_metadata:
            self._steps.append(MetadataExtractor())

    def process(self, documents: list[Document]) -> list[Document]:
        """Run all preprocessing steps on documents."""
        result = documents
        for step in self._steps:
            result = step.process(result)
        return result


def build_preprocessing_pipeline(config: PreprocessingConfig) -> PreprocessingPipeline:
    """Factory function to build a preprocessing pipeline from config."""
    return PreprocessingPipeline(config)