"""Tests for tokenizer abstraction."""

from __future__ import annotations

from rag_sdk.config import TokenizerConfig
from rag_sdk.tokenizer import Cl100kBaseTokenizer, WhitespaceTokenizer, build_tokenizer


def test_whitespace_tokenizer_count():
    """Test whitespace tokenizer count."""
    tokenizer = WhitespaceTokenizer()
    assert tokenizer.count("hello world") == 2
    assert tokenizer.count("") == 0
    assert tokenizer.count("single") == 1
    assert tokenizer.count("multiple   spaces") == 2


def test_whitespace_tokenizer_encode_decode():
    """Test whitespace tokenizer encode/decode."""
    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.encode("hello world")
    assert len(tokens) == 2
    # decode returns placeholder
    decoded = tokenizer.decode(tokens)
    assert "<token>" in decoded


def test_cl100k_base_tokenizer():
    """Test cl100k_base tokenizer if available."""
    try:
        tokenizer = Cl100kBaseTokenizer()
    except ImportError:
        import pytest
        pytest.skip("tiktoken not installed")

    assert tokenizer.count("hello world") > 0
    assert tokenizer.count("") == 0
    tokens = tokenizer.encode("hello world")
    assert len(tokens) == tokenizer.count("hello world")
    decoded = tokenizer.decode(tokens)
    assert decoded == "hello world"


def test_build_tokenizer_whitespace():
    """Test building whitespace tokenizer from config."""
    config = TokenizerConfig(type="whitespace")
    tokenizer = build_tokenizer(config)
    assert isinstance(tokenizer, WhitespaceTokenizer)


def test_build_tokenizer_cl100k_base():
    """Test building cl100k_base tokenizer from config."""
    try:
        config = TokenizerConfig(type="cl100k_base")
        tokenizer = build_tokenizer(config)
        assert isinstance(tokenizer, Cl100kBaseTokenizer)
    except ImportError:
        import pytest
        pytest.skip("tiktoken not installed")


def test_tokenizer_config_defaults():
    """Test tokenizer config defaults."""
    config = TokenizerConfig()
    assert config.type == "whitespace"
    assert config.custom_path is None


def test_tokenizer_config_custom():
    """Test custom tokenizer config."""
    config = TokenizerConfig(type="custom", custom_path="my_module.MyTokenizer")
    assert config.type == "custom"
    assert config.custom_path == "my_module.MyTokenizer"