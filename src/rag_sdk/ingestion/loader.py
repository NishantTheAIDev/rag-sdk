"""Loading source documents from disk.

Phase 2 ships a minimal loader supporting ``.txt``, ``.md``, and ``.json``
corpora. Full document ingestion (PDF, DOCX, HTML) is planned for later phases.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_sdk.core import Document, DocumentMetadata


class IngestionError(ValueError):
    """Raised when a corpus cannot be loaded."""


def load_documents(path: str | Path) -> list[Document]:
    """Load all documents from a directory or a single JSON file.

    ``.txt`` and ``.md`` files each become one document (id is the file stem).
    A ``.json`` file must contain an array of ``{"id", "text", "metadata"}``
    objects.
    """
    source = Path(path)
    if not source.exists():
        raise IngestionError(f"Document path does not exist: {source}")
    if source.is_file():
        return _load_file(source)
    documents: list[Document] = []
    for file in sorted(source.iterdir()):
        if file.suffix.lower() not in {".txt", ".md"}:
            continue
        if not file.is_file():
            continue
        documents.append(_load_text_file(file))
    if not documents:
        raise IngestionError(f"No .txt or .md files found in {source}")
    return documents


def _load_file(path: Path) -> list[Document]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return [_load_text_file(path)]
    raise IngestionError(f"Unsupported document file type: {path.suffix!r}")


def _load_text_file(path: Path) -> Document:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestionError(f"Could not read document: {path}") from exc
    return Document(
        id=path.stem,
        text=text,
        metadata=DocumentMetadata(source=str(path), title=path.stem),
    )


def _load_json(path: Path) -> list[Document]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise IngestionError(f"Could not read document: {path}") from exc
    except json.JSONDecodeError as exc:
        raise IngestionError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise IngestionError(f"{path} must contain a list of documents")
    documents: list[Document] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise IngestionError(f"{path}: entry {index} must be an object")
        text = entry.get("text")
        if not isinstance(text, str) or not text:
            raise IngestionError(f"{path}: entry {index} must have non-empty 'text'")
        doc_id = entry.get("id")
        if doc_id is None:
            doc_id = f"doc-{index}"
        if not isinstance(doc_id, str):
            raise IngestionError(f"{path}: entry {index} 'id' must be a string")
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            raise IngestionError(f"{path}: entry {index} 'metadata' must be an object")
        documents.append(
            Document(
                id=doc_id,
                text=text,
                metadata=DocumentMetadata.model_validate(metadata),
            )
        )
    if not documents:
        raise IngestionError(f"No documents found in {path}")
    return documents