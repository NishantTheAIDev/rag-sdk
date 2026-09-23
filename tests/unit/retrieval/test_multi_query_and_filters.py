"""Tests for multi-query retrieval and metadata filtering across retrievers."""

from __future__ import annotations

import pytest

from rag_sdk.config import (
    BM25RetrievalConfig,
    DenseRetrievalConfig,
    HybridRetrievalConfig,
    MMRRetrievalConfig,
    MultiQueryConfig,
    RagConfig,
)
from rag_sdk.core import Chunk, DocumentMetadata, matches_filters
from rag_sdk.embeddings import HashEmbeddingProvider
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import (
    BM25Retriever,
    MultiQueryRetriever,
    build_retrieval_pipeline,
    build_retriever,
)
from rag_sdk.retrieval.dense import DenseRetriever


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id=f"c{i}",
            text=f"finance report number {i}",
            document_id=f"d{i}",
            index=0,
            start_char=0,
            end_char=10,
            metadata=DocumentMetadata(category="finance" if i % 2 else "tech"),
        )
        for i in range(12)
    ]


def _finance_ids() -> set[str]:
    return {f"c{i}" for i in range(12) if i % 2}


def _config(retrieval: dict) -> RagConfig:
    return RagConfig.model_validate(
        {
            "project": {"name": "t"},
            "chunking": {"strategy": "recursive"},
            "embedding": {"provider": "hash"},
            "retrieval": retrieval,
        }
    )


class TestMultiQuery:
    def test_template_multi_query_fuses_and_fills_top_k(self) -> None:
        embedding = HashEmbeddingProvider()
        config = _config(
            {
                "strategy": "dense",
                "top_k": 5,
                "multi_query": {
                    "enabled": True,
                    "query_generator": "template",
                    "template": "report about {query}",
                },
            }
        )
        pipeline = build_retrieval_pipeline(
            config, embedding, FaissVectorStore(embedding.dimension), None, None, _chunks()
        )
        results = pipeline.search("finance number 3")
        ids = [r.chunk.id for r in results]
        assert len(ids) == 5
        assert len(set(ids)) == 5
        assert all(r.query == "finance number 3" for r in results)

    def test_sub_queries_use_requested_top_k(self) -> None:
        embedding = HashEmbeddingProvider()
        base = DenseRetriever(embedding, FaissVectorStore(embedding.dimension))
        retriever = MultiQueryRetriever(
            base,
            MultiQueryConfig(
                enabled=True, num_queries=2, query_generator="template", template="x {query}"
            ),
        )
        retriever.add_chunks(_chunks())
        assert len(retriever.search("finance report", 10)) == 10

    def test_weighted_fusion(self) -> None:
        embedding = HashEmbeddingProvider()
        base = DenseRetriever(embedding, FaissVectorStore(embedding.dimension))
        retriever = MultiQueryRetriever(
            base,
            MultiQueryConfig(
                enabled=True,
                query_generator="template",
                template="about {query}",
                fusion_method="weighted",
            ),
        )
        retriever.add_chunks(_chunks())
        results = retriever.search("finance report", 4)
        scores = [r.score for r in results]
        assert len(results) == 4
        assert scores == sorted(scores, reverse=True)

    def test_llm_generator_without_generation_config_fails_loudly(self) -> None:
        embedding = HashEmbeddingProvider()
        config = _config({"strategy": "dense", "multi_query": {"enabled": True}})
        with pytest.raises(ValueError, match="generation"):
            build_retrieval_pipeline(
                config, embedding, FaissVectorStore(embedding.dimension), None, None
            )


class TestMetadataFilters:
    def test_matches_filters_semantics(self) -> None:
        meta = DocumentMetadata(category="finance", emails=["a@x.io"])
        assert matches_filters(meta, {"category": "finance"})
        assert matches_filters(meta, {"category": ["tech", "finance"]})
        assert matches_filters(meta, {"emails": "a@x.io"})
        assert not matches_filters(meta, {"category": "tech"})
        assert not matches_filters(meta, {"missing": 1})
        assert matches_filters(None, {})

    def test_bm25_filters(self) -> None:
        retriever = BM25Retriever(BM25RetrievalConfig(filters={"category": "finance"}))
        retriever.add_chunks(_chunks())
        ids = {r.chunk.id for r in retriever.search("finance report", 5)}
        assert len(ids) == 5
        assert ids <= _finance_ids()

    def test_bm25_returns_top_k_even_when_candidate_k_smaller(self) -> None:
        retriever = BM25Retriever(BM25RetrievalConfig(candidate_k=2))
        retriever.add_chunks(_chunks())
        assert len(retriever.search("finance report", 5)) == 5

    @pytest.mark.parametrize(
        "config",
        [
            DenseRetrievalConfig(filters={"category": "finance"}),
            HybridRetrievalConfig(filters={"category": "finance"}),
            MMRRetrievalConfig(filters={"category": "finance"}),
        ],
    )
    def test_dense_hybrid_mmr_filters(self, config) -> None:
        embedding = HashEmbeddingProvider()
        retriever = build_retriever(config, embedding, FaissVectorStore(embedding.dimension))
        retriever.add_chunks(_chunks())
        ids = [r.chunk.id for r in retriever.search("finance report", 10)]
        assert ids
        assert set(ids) <= _finance_ids()

    def test_faiss_re_adding_ids_does_not_duplicate_rows(self) -> None:
        embedding = HashEmbeddingProvider()
        store = FaissVectorStore(embedding.dimension)
        chunks = _chunks()
        vectors = embedding.embed([c.text for c in chunks])
        store.add([c.id for c in chunks], vectors)
        store.add(
            [c.id for c in chunks],
            vectors,
            {c.id: c.metadata.model_dump() for c in chunks},
        )
        assert len(store) == len(chunks)
        hits = store.search(vectors[1], 20, {"category": "finance"})
        assert len(hits) == 6
        assert len({chunk_id for chunk_id, _ in hits}) == 6
