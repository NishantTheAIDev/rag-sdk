"""Pydantic configuration models.

All pipeline settings are represented as validated, serializable models so
that experiments can be reproduced from a configuration alone.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AnswerMetric = Literal[
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "correctness",
    "citation_accuracy",
]
OptimizationMetric = Literal[
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "answer_relevance",
    "latency_ms",
]


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


class SemanticChunkerConfig(ChunkerConfigBase):
    strategy: Literal["semantic"] = "semantic"
    similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    min_chunk_size: int = Field(default=128, ge=1)
    max_chunk_size: int = Field(default=1024, ge=1)
    embedding_provider: str = "hash"
    embedding_model: str | None = None


class StructureAwareChunkerConfig(ChunkerConfigBase):
    strategy: Literal["structure_aware"] = "structure_aware"
    include_heading_context: bool = True
    max_heading_depth: int = Field(default=3, ge=1)
    split_on_headings: list[str] = Field(
        default_factory=lambda: ["h1", "h2", "h3", "h4", "h5", "h6"]
    )


ChunkerConfig = Annotated[
    RecursiveChunkerConfig
    | FixedTokenChunkerConfig
    | SentenceWindowChunkerConfig
    | ParentChildChunkerConfig
    | SemanticChunkerConfig
    | StructureAwareChunkerConfig,
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
    filters: dict[str, str | int | float | bool | list[str] | list[int]] = Field(
        default_factory=dict
    )
    sentence_window: SentenceWindowExpansionConfig = Field(
        default_factory=lambda: SentenceWindowExpansionConfig()
    )
    parent_child: ParentChildExpansionConfig = Field(
        default_factory=lambda: ParentChildExpansionConfig()
    )
    auto_merging: AutoMergingConfig = Field(
        default_factory=lambda: AutoMergingConfig()
    )
    multi_query: MultiQueryConfig = Field(default_factory=lambda: MultiQueryConfig())
    query_rewriter: QueryRewriterConfig = Field(default_factory=lambda: QueryRewriterConfig())


class MultiQueryConfig(BaseModel):
    """Multi-query retrieval configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    num_queries: int = Field(default=3, ge=1, le=10)
    query_generator: Literal["llm", "template"] = "llm"
    template: str | None = None
    fusion_method: Literal["rrf", "weighted"] = "rrf"


class QueryRewriterConfig(BaseModel):
    """Query rewriting configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    strategy: Literal["llm", "template", "hyde"] = "llm"
    model: str | None = None
    template: str | None = None
    prompt: str | None = None


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


class MMRRetrievalConfig(RetrievalConfigBase):
    """Maximal Marginal Relevance retrieval configuration."""

    strategy: Literal["mmr"] = "mmr"
    lambda_param: float = Field(default=0.5, ge=0.0, le=1.0)


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
    DenseRetrievalConfig | BM25RetrievalConfig | HybridRetrievalConfig | MMRRetrievalConfig,
    Field(discriminator="strategy"),
]


class DocumentLoaderConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: str


class TextLoaderConfig(DocumentLoaderConfigBase):
    strategy: Literal["text"] = "text"


class JSONLoaderConfig(DocumentLoaderConfigBase):
    strategy: Literal["json"] = "json"


class PyPDFLoaderConfig(DocumentLoaderConfigBase):
    strategy: Literal["pypdf"] = "pypdf"
    extract_images: bool = False
    page_chunk_size: int = Field(default=1, ge=1)


class DocxLoaderConfig(DocumentLoaderConfigBase):
    strategy: Literal["docx"] = "docx"
    include_headers_footers: bool = False


class HTMLLoaderConfig(DocumentLoaderConfigBase):
    strategy: Literal["html"] = "html"
    extract_main_content: bool = True
    heading_selectors: list[str] = Field(
        default_factory=lambda: ["h1", "h2", "h3", "h4", "h5", "h6"]
    )


DocumentLoaderConfig = Annotated[
    TextLoaderConfig
    | JSONLoaderConfig
    | PyPDFLoaderConfig
    | DocxLoaderConfig
    | HTMLLoaderConfig,
    Field(discriminator="strategy"),
]


class DocumentsConfig(BaseModel):
    """Source corpus for experiments."""

    model_config = ConfigDict(extra="forbid")

    path: str
    loader: DocumentLoaderConfig = Field(default_factory=TextLoaderConfig)
    recursive: bool = True
    glob_pattern: str = "**/*"


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


class TokenizerConfig(BaseModel):
    """Tokenizer configuration for context budgeting."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["whitespace", "cl100k_base", "custom"] = "whitespace"
    custom_path: str | None = None


class GenerationConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512


class MockGenerationConfig(GenerationConfigBase):
    provider: Literal["mock"] = "mock"
    canned_response: str = "Mock response"


class OpenAIGenerationConfig(GenerationConfigBase):
    provider: Literal["openai"] = "openai"
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None


class AnthropicGenerationConfig(GenerationConfigBase):
    provider: Literal["anthropic"] = "anthropic"
    api_key: str | None = None
    base_url: str | None = None


class OllamaGenerationConfig(GenerationConfigBase):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None


GenerationConfig = Annotated[
    MockGenerationConfig
    | OpenAIGenerationConfig
    | AnthropicGenerationConfig
    | OllamaGenerationConfig,
    Field(discriminator="provider"),
]


class JudgeConfig(BaseModel):
    """LLM judge configuration for answer evaluation."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "mock"
    model: str = "mock"
    temperature: float = 0.0


class AnswerEvaluationConfig(BaseModel):
    """Answer evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    reference_based: bool = True
    llm_judge: JudgeConfig | None = None
    metrics: list[AnswerMetric] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "correctness",
            "citation_accuracy",
        ]
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    answer: AnswerEvaluationConfig = Field(default_factory=AnswerEvaluationConfig)


class OptimizationConfig(BaseModel):
    """Optimization configuration."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: OptimizationMetric = "mrr"
    secondary_metric: OptimizationMetric | None = None
    constraints: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    baseline_run_id: str | None = None


class PreprocessingConfig(BaseModel):
    """Preprocessing pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    normalize_whitespace: bool = True
    cleanup_text: bool = True
    remove_headers: bool = False
    remove_duplicates: bool = False
    extract_metadata: bool = False
    header_footer_similarity: float = Field(default=0.8, ge=0.0, le=1.0)
    duplicate_similarity: float = Field(default=0.95, ge=0.0, le=1.0)


class CaptureConfig(BaseModel):
    """Telemetry capture configuration."""

    model_config = ConfigDict(extra="forbid")

    prompts: bool = False
    responses: bool = False
    retrieved_content: bool = False
    document_content: bool = False


class TelemetryConfig(BaseModel):
    """Telemetry configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    capture: CaptureConfig = Field(default_factory=CaptureConfig)


class CacheConfig(BaseModel):
    """Cache configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    ttl_seconds: int = 3600


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
    generation: GenerationConfig | None = None
    evaluation: EvaluationConfig | None = None
    optimization: OptimizationConfig | None = None
    telemetry: TelemetryConfig | None = None
    caching: CacheConfig | None = None
    preprocessing: PreprocessingConfig | None = None