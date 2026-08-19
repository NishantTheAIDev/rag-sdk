"""Tests for the component registry."""

from __future__ import annotations

import pytest

from rag_sdk.core import Registry


class Base:
    pass


class ImplA(Base):
    pass


class ImplB(Base):
    pass


def test_register_and_get() -> None:
    registry = Registry[Base]()
    registry.register("a", ImplA)
    assert registry.get("a") is ImplA


def test_get_missing_raises() -> None:
    registry = Registry[Base]()
    with pytest.raises(KeyError, match="b"):
        registry.get("b")


def test_duplicate_registration_raises() -> None:
    registry = Registry[Base]()
    registry.register("a", ImplA)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a", ImplB)


def test_decorator() -> None:
    registry = Registry[Base]()

    @registry.decorator("decorated")
    class ImplC(Base):
        pass

    assert registry.get("decorated") is ImplC


def test_names_sorted_and_contains() -> None:
    registry = Registry[Base]()
    registry.register("b", ImplB)
    registry.register("a", ImplA)
    assert registry.names() == ["a", "b"]
    assert "a" in registry
    assert "z" not in registry