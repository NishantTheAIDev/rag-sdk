"""Metadata filter matching shared by retrievers and vector stores."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

FilterValue = str | int | float | bool | list[str] | list[int]
MetadataFilters = Mapping[str, FilterValue]


def matches_filters(
    metadata: BaseModel | Mapping[str, object] | None,
    filters: MetadataFilters | None,
) -> bool:
    """Return ``True`` when ``metadata`` satisfies every filter.

    A list filter value matches when the metadata value is one of its items.
    A list metadata value (e.g. ``emails``) matches when it contains the
    filter value, or shares an item with a list filter value.
    """
    if not filters:
        return True
    if metadata is None:
        return False
    values = metadata.model_dump() if isinstance(metadata, BaseModel) else dict(metadata)
    for key, expected in filters.items():
        if key not in values:
            return False
        actual = values[key]
        allowed = expected if isinstance(expected, list) else [expected]
        if isinstance(actual, list):
            if not any(item in allowed for item in actual):
                return False
        elif actual not in allowed:
            return False
    return True