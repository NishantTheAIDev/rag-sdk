"""Tests for YAML config loading."""

from __future__ import annotations

import pytest

from rag_sdk.config import ConfigError, default_config, dump_config, load_config, parse_config

VALID_YAML = """
project:
  name: test
chunking:
  strategy: fixed
  chunk_size: 128
  overlap: 16
"""


def test_parse_valid_yaml() -> None:
    cfg = parse_config(VALID_YAML)
    assert cfg.project.name == "test"
    assert cfg.chunking.strategy == "fixed"


def test_parse_invalid_yaml_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="YAML"):
        parse_config("chunking: [unclosed")


def test_parse_non_mapping_raises() -> None:
    with pytest.raises(ConfigError, match="mapping"):
        parse_config("- just\n- a\n- list")


def test_parse_invalid_schema_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="Invalid configuration"):
        parse_config("chunking:\n  strategy: bogus\n")


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="read"):
        load_config("does/not/exist.yaml")


def test_dump_and_reload_round_trip(tmp_path) -> None:
    path = tmp_path / "rag.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    cfg = load_config(path)
    dumped = dump_config(cfg)
    assert parse_config(dumped) == cfg


def test_default_config() -> None:
    cfg = default_config()
    assert cfg.chunking.strategy == "recursive"
    assert cfg.embedding.provider == "hash"