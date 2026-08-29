"""Preprocessor factory and registry."""

from __future__ import annotations

from rag_sdk.core import Registry
from rag_sdk.preprocessing.base import Preprocessor
from rag_sdk.preprocessing.cleanup import TextCleanup
from rag_sdk.preprocessing.duplicates import DuplicateDetector
from rag_sdk.preprocessing.headers import HeaderFooterRemover
from rag_sdk.preprocessing.metadata import MetadataExtractor
from rag_sdk.preprocessing.whitespace import WhitespaceNormalizer

# Registry for pluggable preprocessors
preprocessor_registry: Registry[type[Preprocessor]] = Registry()
preprocessor_registry.register("whitespace_normalizer", WhitespaceNormalizer)
preprocessor_registry.register("text_cleanup", TextCleanup)
preprocessor_registry.register("header_footer_remover", HeaderFooterRemover)
preprocessor_registry.register("duplicate_detector", DuplicateDetector)
preprocessor_registry.register("metadata_extractor", MetadataExtractor)


def build_preprocessor(name: str, **kwargs) -> Preprocessor:
    """Resolve and construct a preprocessor by name."""
    preprocessor_type = preprocessor_registry.get(name)
    return preprocessor_type(**kwargs)


register_preprocessor = preprocessor_registry.decorator