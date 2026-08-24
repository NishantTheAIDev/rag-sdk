"""Tests for telemetry."""

from __future__ import annotations

from rag_sdk.telemetry import CaptureConfig, TelemetryConfig, TelemetryContext, TelemetryEvent
from rag_sdk.telemetry.console import ConsoleExporter


def test_telemetry_config_defaults():
    """Test telemetry config defaults."""
    config = TelemetryConfig()
    assert config.enabled is False
    assert config.capture.prompts is False
    assert config.capture.responses is False


def test_telemetry_config_enabled():
    """Test telemetry config with enabled."""
    config = TelemetryConfig(enabled=True, capture=CaptureConfig(prompts=True))
    assert config.enabled is True
    assert config.capture.prompts is True


def test_telemetry_context_disabled():
    """Test telemetry context when disabled."""
    config = TelemetryConfig(enabled=False)
    context = TelemetryContext(config)

    context.emit("test_event", {"key": "value"})
    events = context.get_events()

    assert len(events) == 0


def test_telemetry_context_enabled():
    """Test telemetry context when enabled."""
    config = TelemetryConfig(enabled=True)
    context = TelemetryContext(config)

    context.emit("test_event", {"key": "value"})
    events = context.get_events()

    assert len(events) == 1
    assert events[0].name == "test_event"
    assert events[0].attributes["key"] == "value"


def test_telemetry_context_run_variant():
    """Test telemetry context run and variant tracking."""
    config = TelemetryConfig(enabled=True)
    context = TelemetryContext(config)

    context.set_run("run-1")
    context.set_variant("variant-1")
    context.emit("test_event")

    events = context.get_events()
    assert events[0].attributes["run_id"] == "run-1"
    assert events[0].attributes["variant_id"] == "variant-1"


def test_telemetry_event_to_dict():
    """Test telemetry event serialization."""
    event = TelemetryEvent(name="test", attributes={"key": "value"})
    d = event.to_dict()

    assert d["name"] == "test"
    assert d["attributes"]["key"] == "value"
    assert "timestamp" in d


def test_console_exporter():
    """Test console exporter."""
    import io
    output = io.StringIO()
    exporter = ConsoleExporter(output=output)

    events = [TelemetryEvent(name="test", attributes={"a": 1})]
    exporter.export(events)

    result = output.getvalue()
    assert "test" in result
    assert "a" in result