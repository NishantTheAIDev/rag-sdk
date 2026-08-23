"""Tests for cache abstraction."""

from __future__ import annotations

import time

from rag_sdk.cache import CacheConfig, InMemoryCache


def test_cache_disabled():
    """Test cache when disabled."""
    config = CacheConfig(enabled=False)
    cache = InMemoryCache(config)

    cache.set("key", "value")
    assert cache.get("key") is None


def test_cache_basic():
    """Test basic cache operations."""
    config = CacheConfig(enabled=True)
    cache = InMemoryCache(config)

    cache.set("key", "value")
    assert cache.get("key") == "value"


def test_cache_delete():
    """Test cache delete."""
    config = CacheConfig(enabled=True)
    cache = InMemoryCache(config)

    cache.set("key", "value")
    cache.delete("key")
    assert cache.get("key") is None


def test_cache_clear():
    """Test cache clear."""
    config = CacheConfig(enabled=True)
    cache = InMemoryCache(config)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.clear()
    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_cache_ttl():
    """Test cache TTL expiration."""
    config = CacheConfig(enabled=True, ttl_seconds=1)
    cache = InMemoryCache(config)

    cache.set("key", "value", ttl=1)
    assert cache.get("key") == "value"

    time.sleep(1.1)
    assert cache.get("key") is None


def test_cache_default_ttl():
    """Test cache default TTL from config."""
    config = CacheConfig(enabled=True, ttl_seconds=1)
    cache = InMemoryCache(config)

    cache.set("key", "value")  # Uses default TTL
    assert cache.get("key") == "value"

    time.sleep(1.1)
    assert cache.get("key") is None


def test_cache_custom_ttl_override():
    """Test cache custom TTL override."""
    config = CacheConfig(enabled=True, ttl_seconds=1)
    cache = InMemoryCache(config)

    cache.set("key", "value", ttl=10)  # Override TTL
    assert cache.get("key") == "value"

    time.sleep(1.1)
    # Should still be there since we used ttl=10
    assert cache.get("key") == "value"


def test_cache_make_key():
    """Test cache key creation."""
    config = CacheConfig(enabled=True)
    cache = InMemoryCache(config)

    key = cache._make_key("prefix", "suffix")
    assert key == "prefix:suffix"