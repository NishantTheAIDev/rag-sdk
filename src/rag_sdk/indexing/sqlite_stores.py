"""SQLite implementations of DocumentStore and ChunkStore."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rag_sdk.core import Chunk, Document
from rag_sdk.indexing.stores import ChunkStore, DocumentStore


class SqliteDocumentStore(DocumentStore):
    """SQLite-backed document store."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_document(self, document: Document) -> None:
        metadata_json = document.metadata.model_dump_json()
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (id, text, metadata) VALUES (?, ?, ?)",
            (document.id, document.text, metadata_json),
        )
        self._conn.commit()

    def get_document(self, document_id: str) -> Document | None:
        cursor = self._conn.execute(
            "SELECT id, text, metadata FROM documents WHERE id = ?",
            (document_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        doc_id, text, metadata_json = row
        metadata = Document.model_fields["metadata"].annotation.model_validate_json(metadata_json)
        return Document(id=doc_id, text=text, metadata=metadata)

    def close(self) -> None:
        self._conn.close()


class SqliteChunkStore(ChunkStore):
    """SQLite-backed chunk store for parent chunks."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                document_id TEXT NOT NULL,
                index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_parent_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            metadata_json = chunk.metadata.model_dump_json()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO parent_chunks
                (id, text, document_id, index, start_char, end_char, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.text,
                    chunk.document_id,
                    chunk.index,
                    chunk.start_char,
                    chunk.end_char,
                    metadata_json,
                ),
            )
        self._conn.commit()

    def get_parent_chunks(self, parent_ids: list[str]) -> list[Chunk]:
        if not parent_ids:
            return []
        placeholders = ",".join("?" * len(parent_ids))
        cursor = self._conn.execute(
            f"SELECT id, text, document_id, index, start_char, end_char, metadata "
            f"FROM parent_chunks WHERE id IN ({placeholders})",
            parent_ids,
        )
        rows = cursor.fetchall()
        chunks = []
        for row in rows:
            chunk_id, text, doc_id, index, start_char, end_char, metadata_json = row
            metadata = Chunk.model_fields["metadata"].annotation.model_validate_json(metadata_json)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    document_id=doc_id,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                )
            )
        # Return in the same order as requested
        chunk_map = {c.id: c for c in chunks}
        return [chunk_map[pid] for pid in parent_ids if pid in chunk_map]

    def close(self) -> None:
        self._conn.close()


class InMemoryDocumentStore(DocumentStore):
    """In-memory document store for testing."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self._docs: dict[str, Document] = {}
        if documents:
            for doc in documents:
                self._docs[doc.id] = doc

    def add_document(self, document: Document) -> None:
        self._docs[document.id] = document

    def get_document(self, document_id: str) -> Document | None:
        return self._docs.get(document_id)

    def close(self) -> None:
        pass


class InMemoryChunkStore(ChunkStore):
    """In-memory chunk store for testing."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: dict[str, Chunk] = {}
        if chunks:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk

    def add_parent_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def get_parent_chunks(self, parent_ids: list[str]) -> list[Chunk]:
        return [self._chunks[pid] for pid in parent_ids if pid in self._chunks]

    def close(self) -> None:
        pass