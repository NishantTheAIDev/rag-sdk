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


class SentenceWindowChunkerConfig(ChunkerConfigBase):
    strategy: Literal["sentence_window"] = "sentence_window"
    window_size: int = Field(default=3, ge=1)
    window_overlap: int = Field(default=1, ge=0)


class ParentChildChunkerConfig(ChunkerConfigBase):
    strategy: Literal["parent_child"] = "parent_child"
    parent_chunk_size: int = Field(default=1024, ge=1)
    parent_overlap: int = Field(default=128, ge=0)
    child_chunk_size: int = Field(default=256, ge=1)
    child_overlap: int = Field(default=32, ge=0)


ChunkerConfig = Annotated[
    RecursiveChunkerConfig
    | FixedTokenChunkerConfig
    | SentenceWindowChunkerConfig
    | ParentChildChunkerConfig,
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
    candidate_k: int = Field(default=50, ge=1)
    sentence_window: SentenceWindowExpansionConfig = Field(
        default_factory=lambda: SentenceWindowExpansionConfig()
    )
    parent_child: ParentChildExpansionConfig = Field(
        default_factory=lambda: ParentChildExpansionConfig()
    )
    auto_merging: AutoMergingConfig = Field(
        default_factory=lambda: AutoMergingConfig()
    )


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


class SentenceWindowExpansionConfig(BaseModel):
    """Retrieval-time sentence window expansion (independent of chunking strategy)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    window_size: int = Field(default=3, ge=1)


class ParentChildExpansionConfig(BaseModel):
    """Retrieval-time parent-child expansion."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False


class AutoMergingConfig(BaseModel):
    """Retrieval-time auto-merging of adjacent chunks."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    similarity_threshold: float = Field(default=0.8, ge=0, le=1)
    max_tokens: int = Field(default=512, ge=1)
    max_chunks: int = Field(default=10, ge=1)
    tokenizer: Literal["whitespace", "cl100k_base"] = "whitespace"


RetrievalConfig = Annotated[
    DenseRetrievalConfig | BM25RetrievalConfig | HybridRetrievalConfig,
    Field(discriminator="strategy"),
]


class DocumentsConfig(BaseModel):
    """Source corpus for experiments."""

    model_config = ConfigDict(extra="forbid")

    path: str


class RerankerConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1)


class CrossEncoderRerankerConfig(RerankerConfigBase):
    strategy: Literal["cross_encoder"] = "cross_encoder"
    model: str = "BAAI/bge-reranker-base"
    device: str | None = None


class CohereRerankerConfig(RerankerConfigBase):
    strategy: Literal["cohere"] = "cohere"
    model: str = "rerank-v4.0-fast"
    api_key: str | None = None


class NoRerankerConfig(RerankerConfigBase):
    strategy: Literal["none"] = "none"


RerankerConfig = Annotated[
    CrossEncoderRerankerConfig | CohereRerankerConfig | NoRerankerConfig,
    Field(discriminator="strategy"),
]


class ExperimentConfig(BaseModel):
    """Parameter sweeps for the experiment engine."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    parameters: dict[str, list[int | float | str | bool]] = Field(default_factory=dict)
    k: int = Field(default=10, ge=1)
    primary_metric: Literal[
        "hit_at_k", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map"
    ] = "mrr"
    output_dir: str = "runs"


class RagConfig(BaseModel):
    """Root configuration for a RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    chunking: ChunkerConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    reranker: RerankerConfig | None = None
    documents: DocumentsConfig | None = None
    experiments: ExperimentConfig | None = None