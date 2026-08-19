"""Pydantic configuration models.

All pipeline settings are represented as validated, serializable models so
that experiments can be reproduced from a configuration alone.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "rag-project"


class ChunkerConfigBase(BaseModel):
    chunk_size: int = Field(default=512, ge=1)
    overlap: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def _validate_overlap(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class RecursiveChunkerConfig(ChunkerConfigBase):
    strategy: Literal["recursive"]
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "]
    )


class FixedTokenChunkerConfig(ChunkerConfigBase):
    strategy: Literal["fixed"]
    tokenizer: Literal["whitespace"] = "whitespace"


ChunkerConfig = Annotated[
    RecursiveChunkerConfig | FixedTokenChunkerConfig,
    Field(discriminator="strategy"),
]


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "hash"
    model: str | None = None


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["dense"] = "dense"
    top_k: int = Field(default=5, ge=1)


class RagConfig(BaseModel):
    """Root configuration for a RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    chunking: ChunkerConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)