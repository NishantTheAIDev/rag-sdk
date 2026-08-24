"""Telemetry events and context."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rag_sdk.telemetry.config import TelemetryConfig


@dataclass
class TelemetryEvent:
    """A single telemetry event."""

    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


class TelemetryContext:
    """Thread-local telemetry context for the current experiment run."""

    def __init__(self, config: TelemetryConfig) -> None:
        self._config = config
        self._events: list[TelemetryEvent] = []
        self._current_run_id: str | None = None
        self._current_variant_id: str | None = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def config(self) -> TelemetryConfig:
        return self._config

    def set_run(self, run_id: str) -> None:
        self._current_run_id = run_id

    def set_variant(self, variant_id: str) -> None:
        self._current_variant_id = variant_id

    def emit(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Emit a telemetry event."""
        if not self.enabled:
            return
        attrs = attributes or {}
        if self._current_run_id:
            attrs["run_id"] = self._current_run_id
        if self._current_variant_id:
            attrs["variant_id"] = self._current_variant_id
        event = TelemetryEvent(name=name, attributes=attrs)
        self._events.append(event)

    def get_events(self) -> list[TelemetryEvent]:
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()


# Context variable for the current telemetry context
_telemetry_context: ContextVar[TelemetryContext | None] = ContextVar(
    "telemetry_context", default=None
)


def get_telemetry_context() -> TelemetryContext | None:
    """Get the current telemetry context."""
    return _telemetry_context.get()


def set_telemetry_context(context: TelemetryContext | None) -> None:
    """Set the current telemetry context."""
    _telemetry_context.set(context)