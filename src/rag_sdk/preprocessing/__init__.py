"""Preprocessing pipeline for document ingestion."""

from __future__ import annotations

from rag_sdk.preprocessing.base import Preprocessor
from rag_sdk.preprocessing.cleanup import TextCleanup
from rag_sdk.preprocessing.duplicates import DuplicateDetector
from rag_sdk.preprocessing.factory import (
    build_preprocessor,
    preprocessor_registry,
    register_preprocessor,
)
from rag_sdk.preprocessing.headers import HeaderFooterRemover
from rag_sdk.preprocessing.metadata import MetadataExtractor
from rag_sdk.preprocessing.pipeline import PreprocessingPipeline, build_preprocessing_pipeline
from rag_sdk.preprocessing.whitespace import WhitespaceNormalizer

__all__ = [
    "Preprocessor",
    "WhitespaceNormalizer",
    "TextCleanup",
    "HeaderFooterRemover",
    "DuplicateDetector",
    "MetadataExtractor",
    "PreprocessingPipeline",
    "build_preprocessing_pipeline",
    "preprocessor_registry",
    "build_preprocessor",
    "register_preprocessor",
]