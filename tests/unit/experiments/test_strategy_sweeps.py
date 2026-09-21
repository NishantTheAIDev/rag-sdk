"""Regression tests: every strategy can be swept and run end to end."""

from __future__ import annotations

import pytest

from rag_sdk.chunking import build_chunker
from rag_sdk.config import AutoMergingConfig, ParentChildChunkerConfig, RagConfig
from rag_sdk.core import Chunk, Document
from rag_sdk.experiments.grid import expand_grid
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import AutoMerger, RetrievalResult


def _base() -> RagConfig:
    return RagConfig.model_validate(
        {
            "project": {"name": "t"},
            "chunking": {"strategy": "recursive"},
            "embedding": {"provider": "hash"},
            "retrieval": {"strategy": "dense"},
            "reranker": {"strategy": "none"},
        }
    )


@pytest.mark.parametrize(
    ("path", "values"),
    [
        (
            "chunking.strategy",
            [
                "recursive",
                "fixed",
                "sentence_window",
                "parent_child",
                "semantic",
                "structure_aware",
            ],
        ),
        ("retrieval.strategy", ["dense", "bm25", "hybrid", "mmr"]),
        ("reranker.strategy", ["none", "cross_encoder", "cohere"]),
    ],
)
def test_all_strategies_can_be_swept(path: str, values: list[str]) -> None:
    variants = expand_grid(_base(), {path: values})
    section = path.split(".")[0]
    assert [getattr(v, section).strategy for v in variants] == values


def test_parent_child_chunker_builds_and_chunks() -> None:
    chunker = build_chunker(
        ParentChildChunkerConfig(
            parent_chunk_size=200, parent_overlap=0, child_chunk_size=50, child_overlap=0
        )
    )
    result = chunker.chunk(Document(id="d", text="Sentence one is here. " * 40))
    assert result.parents
    assert result.children


def test_auto_merger_exposes_enricher_expand() -> None:
    store = FaissVectorStore(4)
    merger = AutoMerger(store, AutoMergingConfig(enabled=True))
    chunk = Chunk(id="c", text="t", document_id="d", index=0, start_char=0, end_char=1)
    results = [RetrievalResult(query="q", chunk=chunk, score=1.0)]
    assert merger.expand(results) == merger.merge(results)
