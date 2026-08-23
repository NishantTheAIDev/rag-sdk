"""Tokenizer base classes and configuration."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Tokenizer(Protocol):
    """Protocol for tokenizers used in context budgeting and chunking."""

    def count(self, text: str) -> int:
        """Return the number of tokens in the text."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        ...


class TokenizerConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WhitespaceTokenizerConfig(TokenizerConfigBase):
    type: Literal["whitespace"] = "whitespace"


class Cl100kBaseTokenizerConfig(TokenizerConfigBase):
    type: Literal["cl100k_base"] = "cl100k_base"


class CustomTokenizerConfig(TokenizerConfigBase):
    type: Literal["custom"] = "custom"
    module_path: str = Field(..., description="Import path to tokenizer class")


TokenizerConfig = Annotated[
    WhitespaceTokenizerConfig | Cl100kBaseTokenizerConfig | CustomTokenizerConfig,
    Field(discriminator="type"),
]


class WhitespaceTokenizer:
    """Simple whitespace-based tokenizer (no external dependencies)."""

    def count(self, text: str) -> int:
        return len(text.split())

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        return " ".join(["<token>"] * len(tokens))