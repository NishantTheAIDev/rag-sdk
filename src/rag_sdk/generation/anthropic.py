"""Anthropic generator adapter."""

from __future__ import annotations

from typing import Any

from rag_sdk.generation.base import AnthropicGeneratorConfig, GenerationResponse


class AnthropicGenerator:
    """Anthropic Messages API generator."""

    def __init__(self, config: AnthropicGeneratorConfig) -> None:
        self._config = config
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic is required for AnthropicGenerator. "
                "Install with: uv add 'rag-sdk[generation]'"
            ) from e
        self._client = Anthropic(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        **params: Any,
    ) -> GenerationResponse:
        messages = []
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        response = self._client.messages.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            **params,
        )

        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        usage = response.usage

        return GenerationResponse(
            text=text,
            cited_answer=None,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            model=self._config.model,
            provider=self._config.provider,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def provider_name(self) -> str:
        return self._config.provider