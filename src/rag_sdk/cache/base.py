"""Cache base classes and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class CacheConfig(BaseModel):
    """Cache configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable caching")
    ttl_seconds: int = Field(default=3600, ge=1, description="Time-to-live in seconds")


class Cache(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Any | None:
        ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ...

    def delete(self, key: str) -> None:
        ...

    def clear(self) -> None:
        ...


class CacheBase(ABC):
    """Base class for cache implementations."""

    def __init__(self, config: CacheConfig) -> None:
        self._config = config

    @abstractmethod
    def get(self, key: str) -> Any | None:
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    def _make_key(self, *parts: str) -> str:
        """Create a cache key from parts."""
        return ":".join(parts)