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
    from yaml import SafeDumper

    class QuotedDumper(SafeDumper):
        def represent_data(self, data):
            if isinstance(data, str) and ("\n" in data or "\r" in data):
                return self.represent_scalar(
                    "tag:yaml.org,2002:str", data, style='"'
                )
            return super().represent_data(data)

    return yaml.dump(
        config.model_dump(mode="python", exclude_none=True),
        Dumper=QuotedDumper,
        sort_keys=False,
    )


def default_config() -> RagConfig:
    """Return the default configuration used as the ``rag init`` skeleton."""
    return RagConfig.model_validate(
        {
            "project": {"name": "rag-project"},
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 512,
                "overlap": 64,
                "separators": ["\n\n", "\n", ". ", " "],
            },
            "embedding": {"provider": "hash"},
            "retrieval": {"strategy": "dense", "top_k": 5},
        }
    )


def baseline_config(version: str = "v1") -> RagConfig:
    """Return the baseline configuration for benchmarking.

    The baseline is a fixed, reproducible configuration:
    - Recursive chunking (512 tokens, 64 overlap)
    - Hash embeddings (deterministic, no API key needed)
    - Dense retrieval (top_k=10)
    - No reranker
    - No enrichment
    - Mock generator for answer generation
    - Basic retrieval evaluation
    """
    if version != "v1":
        raise ValueError(f"Unknown baseline version: {version}")
    return RagConfig.model_validate(
        {
            "project": {"name": "rag-baseline"},
            "chunking": {"strategy": "recursive", "chunk_size": 512, "overlap": 64},
            "embedding": {"provider": "hash"},
            "retrieval": {
                "strategy": "dense",
                "top_k": 10,
                "auto_merging": {"tokenizer": "whitespace"},
            },
            "reranker": {"strategy": "none"},
            "generation": {"provider": "mock", "model": "mock"},
            "evaluation": {
                "answer": {
                    "enabled": True,
                    "reference_based": True,
                    "metrics": [
                        "faithfulness",
                        "answer_relevance",
                        "context_precision",
                        "context_recall",
                    ],
                }
            },
        }
    )