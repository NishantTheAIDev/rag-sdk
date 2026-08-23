"""Tests for loading experiment datasets."""

from __future__ import annotations

import pytest

from rag_sdk.experiments import DatasetError, load_queries


def test_load_queries(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query": "cats", "relevant_documents": ["cats"]}\n'
        '{"query": "finance", "relevant_documents": ["finance", "banks"]}\n',
        encoding="utf-8",
    )

    samples = load_queries(path)

    assert len(samples) == 2
    assert samples[0].query == "cats"
    assert samples[0].relevant_documents == ["cats"]
    assert samples[1].relevant_documents == ["finance", "banks"]


def test_load_queries_missing_file(tmp_path) -> None:
    with pytest.raises(DatasetError):
        load_queries(tmp_path / "nope.jsonl")


def test_load_queries_invalid_line(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query": "cats"}\n{"relevant_documents": []}\n', encoding="utf-8")
    with pytest.raises(DatasetError):
        load_queries(path)


def test_load_queries_empty_file(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_queries(path)