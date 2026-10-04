import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
from ingest.load_documents import load_documents_from_paths
from ingest.chunk_documents import chunk_documents
from ingest.registry import add_document
from rag.vectorstore import create_or_load_vectorstore

load_dotenv()


def _attach_metadata(chunks, document_id, document_name, version, metadata):
    for chunk in chunks:
        chunk.metadata.update({
            "document_id": document_id,
            "document_name": document_name,
            "version": version,
            **metadata,
        })
    return chunks


def ingest_files(
    file_paths: list[str],
    metadata: dict[str, str] | None = None,
    version: str = "1.0",
    original_names: list[str] | None = None,
):
    """Convert to Markdown (MarkItDown), chunk, attach metadata, and store documents in Qdrant.

    Args:
        file_paths: Paths to documents to ingest.
        metadata: Arbitrary key-value tags attached to every chunk of every
            document (e.g. department, classification) — caller-defined.
        version: Document version string.
        original_names: Human-readable file names to record as document_name,
            parallel to file_paths. Defaults to each path's own file name —
            pass this when file_paths point at temp files with generated names.
    """
    metadata = metadata or {}
    if original_names is None:
        original_names = [Path(p).name for p in file_paths]

    print(f"Converting to Markdown with MarkItDown: {file_paths}")
    docs = load_documents_from_paths(file_paths)
    if not docs:
        print("⚠️ No documents loaded.")
        return None

    print(f"Splitting {len(docs)} Markdown documents into chunks...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    document_records = []
    for path, doc_name in zip(file_paths, original_names):
        doc_id = str(uuid.uuid4())
        path_chunks = [c for c in chunks if c.metadata.get("source", "") == path]
        if not path_chunks:
            path_chunks = chunks  # single-file fallback
        _attach_metadata(path_chunks, document_id=doc_id, document_name=doc_name, version=version, metadata=metadata)
        document_records.append({"document_id": doc_id, "document_name": doc_name, "chunk_count": len(path_chunks)})

    print(f"Uploading {len(chunks)} chunks to Qdrant...")
    create_or_load_vectorstore(documents=chunks)
    print("✅ Ingestion successfully completed.")

    for record in document_records:
        add_document(
            document_id=record["document_id"],
            document_name=record["document_name"],
            version=version,
            metadata=metadata,
            chunk_count=record["chunk_count"],
        )

    return {"total_chunks": len(chunks), "documents": document_records}


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.embed_and_store <file1> [file2 ...]")
        print("  Env vars: VERSION, METADATA (JSON object string)")
        return

    import json
    import os
    paths = [str(Path(p)) for p in sys.argv[1:]]
    try:
        result = ingest_files(
            paths,
            metadata=json.loads(os.getenv("METADATA", "{}")),
            version=os.getenv("VERSION", "1.0"),
        )
        if result:
            print(f"Total chunks indexed: {result['total_chunks']}")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")


if __name__ == "__main__":
    main()
