"""YAML configuration loading and serialization."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag_sdk.config.models import RagConfig


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""


def load_config(path: str | Path) -> RagConfig:
    """Load and validate a RAG configuration from a YAML file."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file: {config_path}") from exc
    return parse_config(text)


def parse_config(text: str) -> RagConfig:
    """Validate a YAML configuration string."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")
    try:
        return RagConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc


def dump_config(config: RagConfig) -> str:
    """Serialize a configuration to YAML."""
    return yaml.safe_dump(
        config.model_dump(mode="json", exclude_none=True), sort_keys=False
    )


def default_config() -> RagConfig:
    """Return the default configuration used as the ``rag init`` skeleton."""
    return RagConfig.model_validate(
        {
            "project": {"name": "rag-project"},
            "chunking": {"strategy": "recursive", "chunk_size": 512, "overlap": 64},
            "embedding": {"provider": "hash"},
            "retrieval": {"strategy": "dense", "top_k": 5},
        }
    )