"""A small generic component registry.

Strategies and providers register their implementation classes by name so
that configuration can resolve implementations without hardcoding. Custom
implementations can register under a name to extend the SDK.
"""

from __future__ import annotations

from collections.abc import Callable


class Registry[T]:
    """Maps names to implementation classes of type ``T``."""

    def __init__(self) -> None:
        self._items: dict[str, type[T]] = {}

    def register(self, name: str, implementation: type[T]) -> None:
        if name in self._items:
            raise ValueError(f"A {name!r} implementation is already registered")
        self._items[name] = implementation

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"No implementation registered under the name {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def decorator(self, name: str) -> Callable[[type[T]], type[T]]:
        def wrap(implementation: type[T]) -> type[T]:
            self.register(name, implementation)
            return implementation

        return wrap