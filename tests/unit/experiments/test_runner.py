"""Tests for the experiment runner."""

from __future__ import annotations

import pytest

from rag_sdk.config import RagConfig
from rag_sdk.core import Document
from rag_sdk.embeddings import HashEmbeddingProvider
from rag_sdk.experiments import ExperimentConfigWarning, ExperimentRunner, run_experiment
from rag_sdk.experiments import runner as runner_module
from rag_sdk.reranking import NoOpReranker

DOCS = [
    Document(
        id="cats",
        text="Cats are small carnivorous mammals. Cats purr when content. "
        "Kittens are young cats. Cat owners value feline companionship. ",
    ),
    Document(
        id="astronomy",
        text="Astronomy studies celestial objects and the cosmos. Planets orbit "
        "stars. The Milky Way is a barred spiral galaxy. Astronomers use telescopes. ",
    ),
    Document(
        id="finance",
        text="Finance covers banking, interest rates, and investments. Stocks "
        "rise and fall. Portfolios balance risk and return. Savers earn compound interest. ",
    ),
]

BASE = RagConfig.model_validate(
    {
        "chunking": {"strategy": "recursive", "chunk_size": 128, "overlap": 16},
        "retrieval": {"strategy": "dense", "top_k": 5},
    }
)


def _write_dataset(tmp_path, lines: list[str]) -> str:
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _base_with_experiments(experiments: dict) -> RagConfig:
    data = BASE.model_dump(mode="json")
    data["experiments"] = experiments
    return RagConfig.model_validate(data)


def test_run_produces_records(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            '{"query": "kittens are young cats", "relevant_documents": ["cats"]}',
            '{"query": "planets orbit stars", "relevant_documents": ["astronomy"]}',
        ],
    )
    base = _base_with_experiments(
        {"dataset": dataset, "k": 3, "output_dir": str(tmp_path / "out")}
    )

    result = run_experiment(DOCS, base)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.total_chunks > 0
    assert record.metrics["hit_at_k"] == 1.0
    assert record.metrics["mrr"] == 1.0
    assert record.latency_ms.mean_ms >= 0
    assert record.latency_ms.median_ms >= 0
    assert record.dataset_hash
    assert record.embedding.provider == "hash"
    assert record.timestamp is not None


def test_run_sweeps_parameters(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 2,
            "parameters": {
                "retrieval.strategy": ["dense", "bm25", "hybrid"],
                "chunking.chunk_size": [64, 128],
            },
        }
    )

    result = run_experiment(DOCS, base)

    assert len(result.records) == 6
    strategies = {record.config["retrieval"]["strategy"] for record in result.records}
    assert strategies == {"dense", "bm25", "hybrid"}
    chunk_sizes = {record.config["chunking"]["chunk_size"] for record in result.records}
    assert chunk_sizes == {64, 128}


def test_leaderboard_sorted_by_primary_metric(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            '{"query": "kittens cats", "relevant_documents": ["cats"]}',
            '{"query": "stars planets", "relevant_documents": ["astronomy"]}',
        ],
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 3,
            "primary_metric": "recall_at_k",
            "parameters": {
                "retrieval.strategy": ["dense", "hybrid"],
                "chunking.chunk_size": [128, 256],
            },
        }
    )

    result = run_experiment(DOCS, base)
    leaderboard = result.leaderboard()

    values = [
        record.metrics[result.primary_metric] for record in leaderboard
    ]
    assert values == sorted(values, reverse=True)


def test_runner_records_skipped_parameters(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 2,
            "parameters": {
                "retrieval.strategy": ["dense", "hybrid"],
                "retrieval.fusion.method": ["weighted"],
            },
        }
    )

    with pytest.warns():
        result = run_experiment(DOCS, base)

    records = {record.config["retrieval"]["strategy"]: record for record in result.records}
    assert records["dense"].skipped_parameters == ["retrieval.fusion.method"]
    assert records["hybrid"].skipped_parameters == []
    assert records["hybrid"].config["retrieval"]["fusion"]["method"] == "weighted"


def test_missing_dataset_raises(tmp_path) -> None:
    base = _base_with_experiments({"dataset": str(tmp_path / "missing.jsonl")})
    with pytest.raises(FileNotFoundError):
        run_experiment(DOCS, base)


def test_no_experiments_section_raises() -> None:
    with pytest.raises(ValueError):
        run_experiment(DOCS, BASE)


def test_runner_rejects_empty_documents(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ['{"query": "q", "relevant_documents": []}'])
    base = _base_with_experiments({"dataset": dataset})
    with pytest.raises(ValueError):
        ExperimentRunner([], base, base.experiments).run()


def test_document_level_metrics_do_not_depend_on_chunk_count(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            '{"query": "kittens are young cats", "relevant_documents": ["cats"]}',
            '{"query": "planets orbit stars", "relevant_documents": ["astronomy"]}',
        ],
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 5,
            "parameters": {"chunking.overlap": [4], "chunking.chunk_size": [32, 256]},
        }
    )

    result = run_experiment(DOCS, base)

    small, large = sorted(result.records, key=lambda r: r.total_chunks, reverse=True)
    assert small.total_chunks > large.total_chunks
    assert result.relevance_level == "document"
    for record in result.records:
        assert record.metrics["recall_at_k"] == 1.0
        assert record.metrics["mrr"] == 1.0


def test_chunk_level_expands_documents_without_chunk_labels(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens are young cats", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 1,
            "relevance_level": "chunk",
            "parameters": {"chunking.overlap": [4], "chunking.chunk_size": [32]},
        }
    )

    record = run_experiment(DOCS, base).records[0]

    # The cats document spans several chunks, but only one is retrieved at k=1.
    assert record.metrics["recall_at_k"] < 1.0


def test_chunk_level_uses_relevant_chunks(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            '{"query": "kittens are young cats", "relevant_documents": ["cats"], '
            '"relevant_chunks": ["not-a-real-chunk"]}'
        ],
    )
    base = _base_with_experiments(
        {"dataset": dataset, "k": 3, "relevance_level": "chunk"}
    )

    record = run_experiment(DOCS, base).records[0]

    assert record.metrics["hit_at_k"] == 0.0


def test_warns_when_top_k_caps_metrics(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments({"dataset": dataset, "k": 10})

    with pytest.warns(ExperimentConfigWarning, match="top_k"):
        record = run_experiment(DOCS, base).records[0]

    assert any("retrieval.top_k (5) < experiments.k (10)" in w for w in record.warnings)


def test_warns_when_reranker_sees_whole_corpus(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "build_reranker", lambda config: NoOpReranker())
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    data = BASE.model_dump(mode="json")
    data["reranker"] = {"strategy": "cross_encoder"}
    data["experiments"] = {"dataset": dataset, "k": 5}
    base = RagConfig.model_validate(data)

    with pytest.warns(ExperimentConfigWarning, match="candidate_k"):
        record = run_experiment(DOCS, base).records[0]

    assert any("reranker sees the whole corpus" in w for w in record.warnings)


def test_no_warnings_for_consistent_config(tmp_path) -> None:
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments({"dataset": dataset, "k": 5})

    assert run_experiment(DOCS, base).records[0].warnings == []


def test_models_are_reused_across_variants(tmp_path, monkeypatch) -> None:
    built: list[object] = []
    embedded: list[str] = []

    class CountingProvider(HashEmbeddingProvider):
        def embed(self, texts):
            embedded.extend(texts)
            return super().embed(texts)

    def build(config):
        built.append(config)
        return CountingProvider()

    monkeypatch.setattr(runner_module, "build_embedding_provider", build)
    dataset = _write_dataset(
        tmp_path, ['{"query": "kittens", "relevant_documents": ["cats"]}']
    )
    base = _base_with_experiments(
        {
            "dataset": dataset,
            "k": 5,
            "parameters": {"retrieval.strategy": ["dense", "mmr", "hybrid"]},
        }
    )

    result = run_experiment(DOCS, base)

    assert len(result.records) == 3
    assert len(built) == 1
    # Chunks are embedded once, not once per variant; queries are re-embedded
    # every run so search latency stays comparable.
    chunk_texts = [text for text in embedded if text != "kittens"]
    assert len(chunk_texts) == len(set(chunk_texts))
    assert embedded.count("kittens") == 3
