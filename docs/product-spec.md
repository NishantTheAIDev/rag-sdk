# Build an Open-Source Python SDK for Production RAG Experimentation & Evaluation

You are a principal software architect and Python library maintainer.

I want to build an **open-source Python SDK** that helps developers **design, experiment with, evaluate, and deploy production-grade RAG systems** through configuration instead of repetitive boilerplate code.

This is **not another LangChain or LlamaIndex clone**. The SDK's primary purpose is to help developers answer:

> **"Which RAG configuration works best for my data?"**

The SDK should support experimentation first and production deployment second.

---

## Vision

Today, every developer implementing RAG has to repeatedly decide:

* Which chunking strategy?
* What chunk size?
* Which embedding model?
* Dense vs Hybrid retrieval?
* Should I use BM25?
* MMR?
* Parent-child retrieval?
* Sentence window retrieval?
* Which reranker?
* How do I evaluate retrieval quality?

I want this SDK to make those decisions configurable and measurable.

A developer should mainly edit a YAML config file and run experiments.

Example:

```yaml
chunking:
  strategy: recursive
  chunk_size: 512
  overlap: 64

embedding:
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: hybrid
  fusion: rrf
```

Then simply run:

```bash
rag experiment
```

and receive a comparison report.

---

# Core Principles

1. Configuration-first architecture
2. Modular & pluggable components
3. Framework agnostic
4. Strong typing (Pydantic)
5. Production quality
6. Excellent developer experience
7. Easily extensible

---

# Project Architecture

```text
rag-sdk/
│
├── configs/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── chunking/
│   ├── embeddings/
│   ├── indexing/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── evaluation/
│   ├── experiments/
│   ├── telemetry/
│   └── cli/
│
├── notebooks/
├── tests/
└── docs/
```

Every module must expose interfaces so users can replace implementations.

Example:

```python
class Chunker(ABC):
    def chunk(self, document): ...
```

---

# SDK Capabilities

## 1. Document Ingestion

Support:

* PDF
* DOCX
* Markdown
* HTML
* TXT
* JSON

Preserve metadata:

* page
* heading
* section hierarchy
* document id
* source path

---

## 2. Chunking Engine

Implement these strategies:

* Fixed Token
* Recursive
* Semantic
* Sentence
* Sentence Window
* Parent/Child
* Structure-aware
* Auto-merging

Each strategy should be independently configurable.

Example:

```yaml
chunking:
  strategy: semantic
  similarity_threshold: 0.82
```

---

## 3. Embedding Engine

Support providers through adapters.

Initial support:

* Sentence Transformers
* OpenAI
* HuggingFace

Embedding interface should allow custom models.

---

## 4. Vector Index Layer

Initial implementation:

* FAISS

Later:

* Qdrant
* Pinecone
* pgvector
* Milvus

The retrieval logic must remain independent of the vector database.

---

## 5. Retrieval Engine

Implement:

* Dense Retrieval
* BM25
* Hybrid
* Reciprocal Rank Fusion
* Weighted Fusion
* MMR
* Metadata Filtering
* Multi-query retrieval
* Query rewriting hooks

---

## 6. Reranking

Support:

* Cross Encoder
* BGE Reranker
* Cohere Rerank (adapter)

Pipeline:

```text
Retriever
   ↓
Top 50
   ↓
Reranker
   ↓
Top 5
```

---

## 7. Evaluation Framework

Separate retrieval evaluation from answer evaluation.

### Retrieval Metrics

* Hit@K
* Recall@K
* Precision@K
* MRR
* nDCG
* MAP

### Answer Metrics

* Faithfulness
* Answer relevance
* Context precision
* Context recall
* Correctness
* Citation accuracy

Support both LLM-based and reference-based evaluation.

---

## 8. Experiment Engine (Most Important Feature)

Allow parameter sweeps.

Example:

```yaml
experiments:

  chunk_size:
    values: [256,512,768]

  embedding:
    values:
      - bge-base
      - e5-base

  retrieval:
    values:
      - dense
      - hybrid
```

Running:

```bash
rag experiment
```

should automatically execute every combination and generate:

* CSV
* JSON
* Interactive HTML report
* Leaderboard

Example output:

| Config          | Recall@10 | MRR  | Latency |
| --------------- | --------- | ---- | ------- |
| Recursive + BGE | 91%       | 0.84 | 42ms    |
| Sentence + E5   | 95%       | 0.89 | 71ms    |

The SDK should automatically recommend the best configuration.

---

# Configuration System

Everything should be driven through a typed YAML schema.

Example:

```yaml
project:
  name: api-rag

documents:
  path: ./docs

chunking:
  strategy: recursive
  chunk_size: 512

embedding:
  provider: sentence-transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategy: hybrid

reranker:
  enabled: true

evaluation:
  enabled: true
```

Validate configs with Pydantic.

---

# CLI

Design an elegant CLI.

Commands:

```bash
rag init
rag ingest
rag index
rag experiment
rag evaluate
rag optimize
rag serve
rag benchmark
```

Example workflow:

```bash
rag init
rag ingest
rag experiment
rag optimize
rag serve
```

---

# Plugin System

Users must be able to register custom implementations.

Example:

```python
class MyChunker(Chunker):
    ...
```

Config:

```yaml
chunking:
  strategy: custom
  implementation: my_pkg.chunker.MyChunker
```

Avoid hardcoded implementations.

---

# Observability

Include built-in telemetry.

Track:

* Retrieval latency
* Embedding latency
* Reranking latency
* Token usage
* Cache hits
* Experiment history

Design this so Phoenix/OpenTelemetry integrations can be added later.

---

# Engineering Requirements

* Python 3.11+
* Pydantic v2
* Typer CLI
* Async-first where appropriate
* Full type hints
* Ruff
* Pytest
* MkDocs documentation
* GitHub Actions CI
* Semantic versioning

Code quality should resemble mature libraries like FastAPI or Pydantic.

---

# Deliverables

Build this SDK incrementally.

## Phase 1

* Project structure
* Config system
* Chunking interfaces
* Recursive + Fixed chunking
* FAISS
* Dense retrieval
* Evaluation metrics
* CLI

## Phase 2

* Hybrid retrieval
* BM25
* Experiment engine
* Reports

## Phase 3

* Sentence window
* Parent-child
* Auto-merging
* Reranking

## Phase 4

* Answer evaluation
* Observability
* Production runtime

---

# Important Design Philosophy

This SDK should optimize **developer decision-making**, not just document retrieval.

Think of it as **"scikit-learn for RAG pipelines"**:

* interchangeable components
* reproducible experiments
* configuration-driven workflows
* measurable performance
* production-ready outputs

Before writing code, first propose:

1. Overall architecture
2. Package structure
3. Public API
4. Configuration schema
5. Interface design
6. Dependency choices

Only after the architecture is approved should implementation begin.
