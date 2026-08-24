"""Console telemetry exporter."""

from __future__ import annotations

import json
import sys
from typing import Any

from rag_sdk.telemetry.events import TelemetryEvent, get_telemetry_context


class ConsoleExporter:
    """Exports telemetry events to console as JSONL."""

    def __init__(self, output: Any = sys.stdout) -> None:
        self._output = output

    def export(self, events: list[TelemetryEvent] | None = None) -> None:
        """Export events to console."""
        if events is None:
            context = get_telemetry_context()
            if context:
                events = context.get_events()
        if not events:
            return

        for event in events:
            self._output.write(json.dumps(event.to_dict()) + "\n")
        self._output.flush()

    def export_all(self) -> None:
        """Export all events from the current context."""
        context = get_telemetry_context()
        if context:
            self.export(context.get_events())


def emit_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Emit an event to the current telemetry context."""
    context = get_telemetry_context()
    if context:
        context.emit(name, attributes)