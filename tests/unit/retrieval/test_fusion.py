"""Tests for hybrid score fusion."""

from __future__ import annotations

import pytest

from rag_sdk.core import Chunk
from rag_sdk.retrieval import rrf_fuse, weighted_fuse
from rag_sdk.retrieval.base import RetrievalResult

CHUNK_KWARGS = dict(document_id="d0", index=0, start_char=0, end_char=1)


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        chunk=Chunk(id=chunk_id, text=chunk_id, **CHUNK_KWARGS),
        score=score,
    )


def test_rrf_merges_and_dedupes() -> None:
    dense = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]
    lexical = [_result("b", 1.2), _result("d", 0.6), _result("a", 0.5)]

    fused = rrf_fuse(dense, lexical, k=60)

    assert {result.chunk.id for result in fused} == {"a", "b", "c", "d"}
    assert fused[0].chunk.id == "b"
    assert fused[1].chunk.id == "a"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_prefers_item_ranked_high_in_both() -> None:
    dense = [_result("shared", 0.9), _result("dense_only", 0.8)]
    lexical = [_result("shared", 0.4), _result("lex_only", 0.3)]

    fused = rrf_fuse(dense, lexical, k=60)

    assert fused[0].chunk.id == "shared"


def test_weighted_fuse_normalizes_and_weights() -> None:
    dense = [_result("a", 1.0), _result("b", 0.5)]
    lexical = [_result("b", 0.9), _result("c", 0.1)]

    fused = weighted_fuse(dense, lexical, dense_weight=0.7, lexical_weight=0.3)

    assert {result.chunk.id for result in fused} == {"a", "b", "c"}
    scores = {result.chunk.id: result.score for result in fused}
    assert scores["a"] == pytest.approx(0.7 * 1.0)
    assert scores["b"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)
    assert scores["c"] == pytest.approx(0.3 * 0.0)


def test_weighted_fuse_with_empty_sources() -> None:
    fused = weighted_fuse([], [], dense_weight=0.5, lexical_weight=0.5)
    assert fused == []