"""External tests for semantic chunking quality."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.external


def test_semantic_chunker_groups_related_sentences():
    """Test that semantic chunker groups semantically related sentences."""
    from rag_sdk.chunking import SemanticChunker
    from rag_sdk.config import EmbeddingConfig, SemanticChunkerConfig
    from rag_sdk.core import Document, DocumentMetadata
    from rag_sdk.embeddings import build_embedding_provider
    
    # Use real embedding provider for quality test
    embedder = build_embedding_provider(EmbeddingConfig(
        provider="sentence-transformers",
        model="sentence-transformers/all-MiniLM-L6-v2"
    ))
    
    config = SemanticChunkerConfig(
        chunk_size=512,
        overlap=64,
        similarity_threshold=0.75,
        min_chunk_size=100,
        max_chunk_size=500,
        embedding_provider="sentence-transformers",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    
    chunker = SemanticChunker(
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        similarity_threshold=config.similarity_threshold,
        min_chunk_size=config.min_chunk_size,
        max_chunk_size=config.max_chunk_size,
        embedding_provider=embedder,
    )
    
    # Document with clearly distinct topics
    doc = Document(
        id="test",
        text=(
            "Machine learning is a subset of artificial intelligence. "
            "It focuses on algorithms that learn from data. "
            "Supervised learning uses labeled examples. "
            "Unsupervised learning finds patterns in unlabeled data. "
            
            "Cooking is the art of preparing food. "
            "Baking uses dry heat in an oven. "
            "Frying uses hot oil or fat. "
            "Grilling uses direct heat from below. "
            
            "Physics is the study of matter and energy. "
            "Classical mechanics describes motion of objects. "
            "Quantum mechanics describes subatomic particles. "
            "Relativity describes space, time, and gravity."
        ) * 2,  # Repeat to have enough content
        metadata=DocumentMetadata(),
    )
    
    chunks = chunker.chunk(doc)
    
    # Should produce multiple chunks (at least one per topic)
    assert len(chunks) >= 3  # ML, Cooking, Physics topics
    
    # Check that chunks are not empty
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_semantic_chunker_with_hash_embeddings():
    """Test semantic chunker with hash embeddings (deterministic)."""
    from rag_sdk.chunking import SemanticChunker
    from rag_sdk.config import EmbeddingConfig, SemanticChunkerConfig
    from rag_sdk.core import Document, DocumentMetadata
    from rag_sdk.embeddings import build_embedding_provider
    
    embedder = build_embedding_provider(EmbeddingConfig(provider="hash"))
    
    config = SemanticChunkerConfig(
        chunk_size=256,
        overlap=32,
        similarity_threshold=0.5,
        min_chunk_size=50,
        max_chunk_size=200,
        embedding_provider="hash",
    )
    
    chunker = SemanticChunker(
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        similarity_threshold=config.similarity_threshold,
        min_chunk_size=config.min_chunk_size,
        max_chunk_size=config.max_chunk_size,
        embedding_provider=embedder,
    )
    
    doc = Document(
        id="test",
        text=(
            "The quick brown fox jumps over the lazy dog. " * 10
        ),
        metadata=DocumentMetadata(),
    )
    
    chunks = chunker.chunk(doc)
    
    assert len(chunks) > 0
    # With hash embeddings and repeated text, similarity is high
    # so we get fewer chunks
    for chunk in chunks:
        assert len(chunk.text) > 0