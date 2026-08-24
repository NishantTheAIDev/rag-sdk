"""Tokenizer factory resolved by configuration."""

from __future__ import annotations

from importlib import import_module

from rag_sdk.core import Registry
from rag_sdk.tokenizer import Tokenizer, TokenizerConfig
from rag_sdk.tokenizer.base import WhitespaceTokenizer
from rag_sdk.tokenizer.tiktoken_adapter import Cl100kBaseTokenizer

tokenizer_registry: Registry[type[Tokenizer]] = Registry()
tokenizer_registry.register("whitespace", WhitespaceTokenizer)
tokenizer_registry.register("cl100k_base", Cl100kBaseTokenizer)


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """Construct a tokenizer from its configuration."""
    match config:
        case config if config.type == "whitespace":
            return WhitespaceTokenizer()
        case config if config.type == "cl100k_base":
            return Cl100kBaseTokenizer()
        case config if config.type == "custom":
            module_path = config.module_path
            module_name, class_name = module_path.rsplit(".", 1)
            module = import_module(module_name)
            tokenizer_class = getattr(module, class_name)
            return tokenizer_class()
        case _:  # pragma: no cover
            raise ValueError(f"Unsupported tokenizer type: {config.type!r}")


def register_tokenizer(name: str, tokenizer_class: type[Tokenizer]) -> None:
    """Register a custom tokenizer."""
    tokenizer_registry.register(name, tokenizer_class)