"""Tokenizer abstraction for token counting and encoding."""

from __future__ import annotations

from rag_sdk.tokenizer.base import Tokenizer, TokenizerConfig, WhitespaceTokenizer
from rag_sdk.tokenizer.factory import build_tokenizer, tokenizer_registry
from rag_sdk.tokenizer.tiktoken_adapter import Cl100kBaseTokenizer

__all__ = [
    "Tokenizer",
    "TokenizerConfig",
    "WhitespaceTokenizer",
    "Cl100kBaseTokenizer",
    "build_tokenizer",
    "tokenizer_registry",
    "register_tokenizer",
]

register_tokenizer = tokenizer_registry.decorator