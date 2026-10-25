import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(os.getenv("DOCUMENT_REGISTRY_PATH", Path(__file__).resolve().parent.parent / "data" / "documents_registry.json"))
_lock = threading.Lock()


def _read() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(entries: list[dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def add_document(
    document_id: str,
    document_name: str,
    version: str,
    chunk_count: int,
    metadata: dict[str, str] | None = None,
    pipeline: str = "standard",
    chunking: bool = False,
    captioning: bool = False,
    models: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Record an indexed document.

    `pipeline` names the ingestion pipeline ("standard" or "docling"); `chunking` and
    `captioning` record which techniques it was indexed with — structure-aware (Docling
    HybridChunker) chunking and VLM figure descriptions — so an evaluation run can tell
    the configurations of one document apart. `models` names
    the LLM behind each LLM-driven technique ({"captioning": ...}). `embedding`
    records the embedding model, dimension and Qdrant collection the chunks went into.
    """
    from rag.vectorstore import EMBEDDING_DIMENSION, QDRANT_COLLECTION_NAME, embedding_model_name

    entry = {
        "document_id": document_id,
        "document_name": document_name,
        "version": version,
        "chunk_count": chunk_count,
        "metadata": metadata or {},
        "pipeline": pipeline,
        "chunking": chunking,
        "captioning": captioning,
        "models": models or {},
        "embedding": {
            "model": embedding_model_name(),
            "dimension": EMBEDDING_DIMENSION,
            "collection": QDRANT_COLLECTION_NAME,
        },
        "status": "indexed",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        entries = _read()
        entries.append(entry)
        _write(entries)
    return entry


def list_documents() -> list[dict[str, Any]]:
    with _lock:
        return sorted(_read(), key=lambda e: e["uploaded_at"], reverse=True)


def get_document(document_id: str) -> dict[str, Any] | None:
    for entry in list_documents():
        if entry["document_id"] == document_id:
            return entry
    return None


def remove_document(document_id: str) -> bool:
    with _lock:
        entries = _read()
        remaining = [e for e in entries if e["document_id"] != document_id]
        if len(remaining) == len(entries):
            return False
        _write(remaining)
        return True
