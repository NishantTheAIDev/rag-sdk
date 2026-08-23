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
    dimension: int | None = Field(default=None, ge=1)


class RetrievalConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1)


class DenseRetrievalConfig(RetrievalConfigBase):
    strategy: Literal["dense"] = "dense"


class BM25Params(BaseModel):
    """Okapi BM25 parameters shared by the BM25 and hybrid strategies."""

    model_config = ConfigDict(extra="forbid")

    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    tokenizer: Literal["default", "whitespace"] = "default"
    stopwords: bool = True


class BM25RetrievalConfig(BM25Params, RetrievalConfigBase):
    strategy: Literal["bm25"] = "bm25"


class FusionConfig(BaseModel):
    """How dense and lexical scores are combined for hybrid retrieval."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.5, ge=0, le=1)
    bm25_weight: float = Field(default=0.5, ge=0, le=1)
    candidate_k: int = Field(default=50, ge=1)

    @model_validator(mode="after")
    def _validate_weights(self) -> Self:
        if self.method == "weighted" and self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("weighted fusion requires positive combined weights")
        return self


class HybridRetrievalConfig(RetrievalConfigBase):
    strategy: Literal["hybrid"] = "hybrid"
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    bm25: BM25Params = Field(default_factory=BM25Params)


RetrievalConfig = Annotated[
    DenseRetrievalConfig | BM25RetrievalConfig | HybridRetrievalConfig,
    Field(discriminator="strategy"),
]


class DocumentsConfig(BaseModel):
    """Source corpus for experiments."""

    model_config = ConfigDict(extra="forbid")

    path: str


class ExperimentConfig(BaseModel):
    """Parameter sweeps for the experiment engine."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    parameters: dict[str, list[int | float | str | bool]] = Field(default_factory=dict)
    k: int = Field(default=10, ge=1)
    primary_metric: Literal[
        "hit_at_k", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map"
    ] = "mrr"
    output_dir: str = "experiments"


class RagConfig(BaseModel):
    """Root configuration for a RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    chunking: ChunkerConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    documents: DocumentsConfig | None = None
    experiments: ExperimentConfig | None = None