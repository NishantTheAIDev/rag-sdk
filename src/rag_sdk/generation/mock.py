"""Mock generator for deterministic testing."""

from __future__ import annotations

from typing import Any

from rag_sdk.generation.base import GenerationResponse, MockGeneratorConfig


class MockGenerator:
    """Deterministic generator returning canned responses."""

    def __init__(self, config: MockGeneratorConfig) -> None:
        self._config = config

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        **params: Any,
    ) -> GenerationResponse:
        return GenerationResponse(
            text=self._config.canned_response,
            cited_answer=None,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self._config.canned_response.split()),
            total_tokens=len(prompt.split()) + len(self._config.canned_response.split()),
            model=self._config.model,
            provider=self._config.provider,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def provider_name(self) -> str:
        return self._config.provider