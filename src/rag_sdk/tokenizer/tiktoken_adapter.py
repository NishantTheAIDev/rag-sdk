"""Tiktoken-based tokenizer adapter (optional dependency)."""

from __future__ import annotations


class Cl100kBaseTokenizer:
    """OpenAI's cl100k_base tokenizer via tiktoken."""

    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for Cl100kBaseTokenizer. "
                "Install with: uv add 'rag-sdk[tokenizers]'"
            ) from e
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)