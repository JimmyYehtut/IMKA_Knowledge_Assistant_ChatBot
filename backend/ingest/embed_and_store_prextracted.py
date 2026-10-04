"""
Pre-extracted ingestion pipeline: loads already-chunked document knowledge
(an entities.jsonl produced by an earlier Docling extraction run) together
with per-page diagram images extracted from the same PDF, and attaches each
image to the chunk whose text best describes it, so the diagram surfaces
next to the citation that mentions it at query time.

This is a stopgap for documents whose images were captured without VLM
captions. See embed_and_store_docling.py's caption_figures option for the
proper long-term path (each figure gets a purpose-written description);
here we reuse each image's surrounding text as its retrievable
"description" instead, since none exists yet for this dataset.

Usage: python -m ingest.embed_and_store_prextracted <entities.jsonl> <images_dir> <doc_name> [version]
"""
import json
import re
import shutil
import sys
import uuid
from pathlib import Path

from langchain_core.documents import Document

from ingest.registry import add_document
from rag.vectorstore import create_or_load_vectorstore

DIAGRAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "diagrams"

_IMAGE_NAME_RE = re.compile(r"p(\d+)_img\d+\.\w+$", re.IGNORECASE)
_FIGURE_REF_RE = re.compile(r"Figure \d+")


def _load_chunks(entities_path: Path) -> list[dict]:
    chunks = []
    with open(entities_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _pick_best_chunk_for_image(page: int, chunks: list[dict]) -> tuple[dict | None, str]:
    """Heuristic: prefer the chunk confined to exactly this page (a table, if
    any, since these images are almost always rasterized tables), else a
    chunk whose text references "Figure N", else the first chunk whose page
    range covers this page. Returns (chunk, confidence)."""
    candidates = [c for c in chunks if c.get("page_start") is not None and c["page_start"] <= page <= c["page_end"]]

    single_page = [c for c in candidates if c["page_start"] == c["page_end"] == page]
    if single_page:
        tables = [c for c in single_page if c.get("chunk_type") == "table"]
        return (tables[0] if tables else single_page[0]), "high"

    figure_ref = [c for c in candidates if _FIGURE_REF_RE.search(c["text"])]
    if figure_ref:
        return figure_ref[0], "medium"

    if candidates:
        return candidates[0], "low"

    return None, "none"


def _entity_types(chunk: dict) -> list[str]:
    return sorted({e["type"] for e in chunk.get("entities", [])})


def ingest_prextracted(
    entities_path: str,
    images_dir: str,
    doc_name: str,
    version: str = "1.0",
    metadata: dict[str, str] | None = None,
) -> dict | None:
    """Ingest a document from a pre-extracted entities.jsonl plus a folder of
    page-indexed diagram images (filenames like p012_img01.png).

    Every chunk in entities.jsonl becomes a retrievable Document, reusing the
    entities already extracted for it as metadata. Each image is matched to
    whichever chunk best describes it (_pick_best_chunk_for_image) and its
    path is copied under data/diagrams/<document_id>/ and recorded on that
    chunk's metadata as image_path, so it survives into citations built by
    rag.pipeline._stage6_build_citations.
    """
    metadata = metadata or {}
    entities_path = Path(entities_path)
    images_dir = Path(images_dir)

    raw_chunks = _load_chunks(entities_path)
    if not raw_chunks:
        print("⚠️ No chunks found in entities file.")
        return None

    document_id = str(uuid.uuid4())
    dest_dir = DIAGRAMS_DIR / document_id

    image_by_chunk_id: dict[str, tuple[Path, str]] = {}
    image_files = sorted(images_dir.glob("*")) if images_dir.exists() else []
    for image_path in image_files:
        m = _IMAGE_NAME_RE.match(image_path.name)
        if not m:
            continue
        page = int(m.group(1))
        best_chunk, confidence = _pick_best_chunk_for_image(page, raw_chunks)
        if best_chunk is None:
            print(f"  [warn] no chunk found for {image_path.name} (page {page}), skipping image", file=sys.stderr)
            continue
        image_by_chunk_id[best_chunk["chunk_id"]] = (image_path, confidence)

    documents: list[Document] = []
    for chunk in raw_chunks:
        doc_meta = {
            "source": str(entities_path),
            "document_id": document_id,
            "document_name": doc_name,
            "version": version,
            "page_no": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "section_path": chunk.get("section_path", ""),
            "chunk_type": chunk.get("chunk_type", "prose"),
            "entity_types": _entity_types(chunk),
            **metadata,
        }

        match = image_by_chunk_id.get(chunk["chunk_id"])
        if match:
            image_path, confidence = match
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_dir / image_path.name)
            doc_meta["image_path"] = f"diagrams/{document_id}/{image_path.name}"
            doc_meta["image_match_confidence"] = confidence

        documents.append(Document(page_content=chunk["text"], metadata=doc_meta))

    diagram_count = sum(1 for d in documents if "image_path" in d.metadata)
    print(f"Uploading {len(documents)} chunks ({diagram_count} with diagrams) to Qdrant...")
    create_or_load_vectorstore(documents=documents)
    print("✅ Pre-extracted ingestion successfully completed.")

    add_document(
        document_id=document_id,
        document_name=doc_name,
        version=version,
        metadata=metadata,
        chunk_count=len(documents),
    )

    return {
        "document_id": document_id,
        "document_name": doc_name,
        "total_chunks": len(documents),
        "diagram_chunks": diagram_count,
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m ingest.embed_and_store_prextracted <entities.jsonl> <images_dir> <doc_name> [version]")
        return

    entities_path, images_dir, doc_name = sys.argv[1:4]
    version = sys.argv[4] if len(sys.argv) > 4 else "1.0"

    try:
        result = ingest_prextracted(entities_path, images_dir, doc_name, version=version)
        if result:
            print(
                f"Indexed {result['total_chunks']} chunks "
                f"({result['diagram_chunks']} with diagrams) for "
                f"{result['document_name']} ({result['document_id']})"
            )
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")


if __name__ == "__main__":
    main()
