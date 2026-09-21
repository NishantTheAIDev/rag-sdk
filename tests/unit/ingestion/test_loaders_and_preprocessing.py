"""Tests for configured document loading and preprocessing wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_sdk.config import HTMLLoaderConfig, PreprocessingConfig, TextLoaderConfig
from rag_sdk.core import Document
from rag_sdk.ingestion import HTMLLoader, IngestionError, load_documents
from rag_sdk.preprocessing import DuplicateDetector, build_preprocessing_pipeline

HTML = """<html><head><title>Guide</title></head><body>
<nav>menu</nav>
<article><h1>Guide</h1><p>Retrieval augmented generation combines search with
language models so answers are grounded in documents. It is widely used.</p>
<p>Chunking splits documents into passages before they are embedded.</p></article>
</body></html>"""


def test_html_loader_returns_plain_text(tmp_path: Path) -> None:
    pytest.importorskip("bs4")
    page = tmp_path / "page.html"
    page.write_text(HTML, encoding="utf-8")
    for config in (None, HTMLLoaderConfig(extract_main_content=False)):
        text = HTMLLoader().load(page, config)[0].text
        assert "Retrieval augmented generation" in text
        assert "<" not in text and ">" not in text


def test_configured_loader_scans_directories(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.md").write_text("beta", encoding="utf-8")
    (nested / "skip.json").write_text("[]", encoding="utf-8")

    flat = load_documents(tmp_path, TextLoaderConfig(), recursive=False, glob_pattern="**/*")
    assert [d.id for d in flat] == ["a"]
    deep = load_documents(tmp_path, TextLoaderConfig(), recursive=True, glob_pattern="**/*")
    assert sorted(d.id for d in deep) == ["a", "b"]


def test_configured_loader_errors_when_no_files_match(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    with pytest.raises(IngestionError, match=".html"):
        load_documents(tmp_path, HTMLLoaderConfig(), recursive=True, glob_pattern="**/*")


def test_duplicate_detector_removes_near_duplicates() -> None:
    documents = [
        Document(id="a", text="The quick brown fox jumps over the lazy dog."),
        Document(id="b", text="The quick brown fox jumps over the lazy dog!"),
        Document(id="c", text="Completely different content about retrieval."),
    ]
    assert [d.id for d in DuplicateDetector(similarity_threshold=0.9).process(documents)] == [
        "a",
        "c",
    ]
    pipeline = build_preprocessing_pipeline(
        PreprocessingConfig(remove_duplicates=True, duplicate_similarity=0.9)
    )
    assert [d.id for d in pipeline.process(documents)] == ["a", "c"]
