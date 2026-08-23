"""OpenAI generator adapter."""

from __future__ import annotations

from typing import Any

from rag_sdk.generation.base import GenerationResponse, OpenAIGeneratorConfig


class OpenAIGenerator:
    """OpenAI chat completions generator."""

    def __init__(self, config: OpenAIGeneratorConfig) -> None:
        self._config = config
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIGenerator. "
                "Install with: uv add 'rag-sdk[generation]'"
            ) from e
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            organization=config.organization,
        )

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        **params: Any,
    ) -> GenerationResponse:
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            **params,
        )

        choice = response.choices[0]
        usage = response.usage

        return GenerationResponse(
            text=choice.message.content or "",
            cited_answer=None,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=self._config.model,
            provider=self._config.provider,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def provider_name(self) -> str:
        return self._config.provider