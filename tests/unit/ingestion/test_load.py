"""Tests for the minimal document loader."""

from __future__ import annotations

import json

import pytest

from rag_sdk.ingestion import IngestionError, load_documents


def test_loads_text_and_markdown_directory(tmp_path) -> None:
    (tmp_path / "cats.md").write_text("Cats are mammals.", encoding="utf-8")
    (tmp_path / "finance.txt").write_text("Interest rates rise.", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert {doc.id for doc in documents} == {"cats", "finance"}
    assert {doc.metadata.title for doc in documents} == {"cats", "finance"}
    assert all(doc.text for doc in documents)


def test_loads_json_document_array(tmp_path) -> None:
    path = tmp_path / "docs.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a", "text": "hello", "metadata": {"title": "A"}},
                {"id": "b", "text": "world"},
            ]
        ),
        encoding="utf-8",
    )

    documents = load_documents(path)

    assert len(documents) == 2
    assert documents[0].metadata.title == "A"
    assert documents[1].id == "b"


def test_missing_path_raises(tmp_path) -> None:
    with pytest.raises(IngestionError):
        load_documents(tmp_path / "nope")


def test_empty_directory_raises(tmp_path) -> None:
    with pytest.raises(IngestionError):
        load_documents(tmp_path)


def test_invalid_json_raises(tmp_path) -> None:
    path = tmp_path / "docs.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(IngestionError):
        load_documents(path)