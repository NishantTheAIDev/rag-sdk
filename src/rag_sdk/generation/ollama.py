"""Ollama generator adapter for local models."""

from __future__ import annotations

from typing import Any

import httpx

from rag_sdk.generation.base import GenerationResponse, OllamaGeneratorConfig


class OllamaGenerator:
    """Ollama local model generator via HTTP API."""

    def __init__(self, config: OllamaGeneratorConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=120.0,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else None,
        )

    def generate(
        self,
        prompt: str,
        context: str | None = None,
        **params: Any,
    ) -> GenerationResponse:
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"

        payload = {
            "model": self._config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
                **params,
            },
        }

        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        text = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return GenerationResponse(
            text=text,
            cited_answer=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self._config.model,
            provider=self._config.provider,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def provider_name(self) -> str:
        return self._config.provider