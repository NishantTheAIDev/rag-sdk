"""End-to-end pipeline: config -> chunk -> embed -> index -> retrieve -> evaluate.

Runs entirely in memory with the deterministic hash embedding fixture; no
network or API keys required.
"""

from __future__ import annotations

from pathlib import Path

from rag_sdk.chunking import build_chunker
from rag_sdk.config import load_config
from rag_sdk.core import Chunk, Document
from rag_sdk.evaluation import evaluate_retrieval
from rag_sdk.indexing import FaissVectorStore
from rag_sdk.retrieval import DenseRetriever

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "example.yaml"

DOCS = [
    Document(
        id="cats",
        text=(
            "Cats are small carnivorous mammals. Cats purr when content. "
            "Kittens are young cats. Cat owners value feline companionship. "
        ) * 6,
        metadata={"source": "cats.md", "title": "Cats"},
    ),
    Document(
        id="astronomy",
        text=(
            "Astronomy studies celestial objects and the cosmos. Planets orbit stars. "
            "The Milky Way is a barred spiral galaxy. Astronomers use telescopes. "
        ) * 6,
        metadata={"source": "astronomy.md", "title": "Astronomy"},
    ),
    Document(
        id="finance",
        text=(
            "Finance covers banking, interest rates, and investments. Stocks rise and fall. "
            "Portfolios balance risk and return. Savers earn compound interest. "
        ) * 6,
        metadata={"source": "finance.md", "title": "Finance"},
    ),
]


def _chunk_corpus() -> list[Chunk]:
    config = load_config(CONFIG_PATH)
    chunker = build_chunker(config.chunking)
    chunks = [chunk for doc in DOCS for chunk in chunker.chunk(doc)]
    assert len(chunks) >= 6
    assert {chunk.document_id for chunk in chunks} == {"cats", "astronomy", "finance"}
    return chunks


def test_pipeline_retrieves_relevant_document(hash_embedding) -> None:
    chunks = _chunk_corpus()
    store = FaissVectorStore(dimension=hash_embedding.dimension)
    retriever = DenseRetriever(hash_embedding, store)
    retriever.add_chunks(chunks)

    results = retriever.search("kittens purring felines", top_k=5)

    assert results
    top_ids = {result.chunk.id for result in results}
    assert any(chunk_id.startswith("cats:") for chunk_id in top_ids)


def test_pipeline_evaluates_metrics(hash_embedding) -> None:
    chunks = _chunk_corpus()
    store = FaissVectorStore(dimension=hash_embedding.dimension)
    retriever = DenseRetriever(hash_embedding, store)
    retriever.add_chunks(chunks)

    queries = {
        "cats": "kittens are young cats",
        "astronomy": "planets orbit stars",
        "finance": "savers earn compound interest",
    }
    samples = []
    for document_id, query in queries.items():
        retrieved = [result.chunk.id for result in retriever.search(query, top_k=3)]
        relevant = {chunk.id for chunk in chunks if chunk.document_id == document_id}
        samples.append((retrieved, relevant))

    metrics = evaluate_retrieval(samples, k=3)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["recall_at_k"] > 0.0
    assert metrics["mrr"] == 1.0
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert metrics["ndcg_at_k"] == 1.0


def test_example_config_is_valid() -> None:
    config = load_config(CONFIG_PATH)
    assert config.chunking.strategy == "recursive"
    assert config.project.name == "api-rag"