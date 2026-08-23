"""Telemetry configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CaptureConfig(BaseModel):
    """What content to capture in telemetry."""

    model_config = ConfigDict(extra="forbid")

    prompts: bool = Field(default=False, description="Capture prompt text")
    responses: bool = Field(default=False, description="Capture response text")
    retrieved_content: bool = Field(default=False, description="Capture retrieved chunk text")
    document_content: bool = Field(default=False, description="Capture document text")


class TelemetryConfig(BaseModel):
    """Telemetry configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable telemetry")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)