"""Tests for the FAISS vector store."""

from __future__ import annotations

import numpy as np
import pytest

from rag_sdk.indexing import FaissVectorStore


def _normalized(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_add_and_search_round_trip() -> None:
    store = FaissVectorStore(dimension=4)
    vectors = _normalized(np.eye(4, dtype=np.float32))
    store.add(["a", "b", "c", "d"], vectors)
    assert len(store) == 4
    hits = store.search(vectors[2], k=2)
    assert hits[0][0] == "c"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_returns_top_k_sorted() -> None:
    store = FaissVectorStore(dimension=2)
    store.add(
        ["x", "y"],
        _normalized(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
    )
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), k=5)
    assert [h[0] for h in hits] == ["x", "y"]


def test_search_on_empty_store() -> None:
    store = FaissVectorStore(dimension=3)
    assert store.search(np.zeros(3, dtype=np.float32), k=3) == []


def test_k_less_than_one_returns_empty() -> None:
    store = FaissVectorStore(dimension=2)
    store.add(["a"], _normalized(np.array([[1.0, 0.0]], dtype=np.float32)))
    assert store.search(np.array([1.0, 0.0], dtype=np.float32), k=0) == []


def test_wrong_dimension_rejected() -> None:
    store = FaissVectorStore(dimension=2)
    with pytest.raises(ValueError, match="shape"):
        store.add(["a"], np.zeros((1, 3), dtype=np.float32))
    store.add(["a"], _normalized(np.array([[1.0, 0.0]], dtype=np.float32)))
    with pytest.raises(ValueError, match="dimensions"):
        store.search(np.zeros(3, dtype=np.float32), k=1)


def test_duplicate_ids_rejected() -> None:
    store = FaissVectorStore(dimension=2)
    with pytest.raises(ValueError, match="unique"):
        store.add(
            ["a", "a"],
            _normalized(np.eye(2, dtype=np.float32)),
        )


def test_mismatched_ids_and_vectors_rejected() -> None:
    store = FaissVectorStore(dimension=2)
    with pytest.raises(ValueError, match="same length"):
        store.add(["a"], _normalized(np.eye(2, dtype=np.float32)))