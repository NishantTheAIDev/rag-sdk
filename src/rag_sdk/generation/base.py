"""Generation base classes and configuration."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """Citation linking answer text to source chunks."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    source_uri: str | None = None
    text_span: tuple[int, int] | None = None
    score: float = 0.0


class CitedAnswer(BaseModel):
    """Answer with inline citations."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    """Response from a generation provider."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text: str
    cited_answer: CitedAnswer | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Generator(Protocol):
    """Protocol for generation providers."""

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        **params: Any,
    ) -> GenerationResponse:
        """Generate a response for the given prompt and optional context."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        ...


class GenerationConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)


class MockGeneratorConfig(GenerationConfigBase):
    provider: Literal["mock"] = "mock"
    canned_response: str = "This is a mock response for testing."


class OpenAIGeneratorConfig(GenerationConfigBase):
    provider: Literal["openai"] = "openai"
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None


class AnthropicGeneratorConfig(GenerationConfigBase):
    provider: Literal["anthropic"] = "anthropic"
    api_key: str | None = None
    base_url: str | None = None


class OllamaGeneratorConfig(GenerationConfigBase):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None


GenerationConfig = Annotated[
    MockGeneratorConfig
    | OpenAIGeneratorConfig
    | AnthropicGeneratorConfig
    | OllamaGeneratorConfig,
    Field(discriminator="provider"),
]