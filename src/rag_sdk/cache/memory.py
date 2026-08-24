"""In-memory cache implementation."""

from __future__ import annotations

import time
from typing import Any

from rag_sdk.cache.base import CacheBase, CacheConfig


class CacheEntry:
    """A cache entry with expiration."""

    def __init__(self, value: Any, expires_at: float | None = None) -> None:
        self.value = value
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class InMemoryCache(CacheBase):
    """Simple in-memory cache with TTL support."""

    def __init__(self, config: CacheConfig) -> None:
        super().__init__(config)
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        if not self._config.enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self._config.enabled:
            return
        ttl = ttl or self._config.ttl_seconds
        expires_at = time.time() + ttl if ttl > 0 else None
        self._store[key] = CacheEntry(value, expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        expired_keys = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired_keys:
            del self._store[k]
        return len(expired_keys)