"""
Docling-based ingestion pipeline (alternative to ingest/embed_and_store.py).

Parses PDFs with Docling (layout + table structure aware), chunks them with
Docling's HybridChunker (header-aware, table-atomic, token-budget aware, tables
serialized as Markdown), saves each captioned figure's image under
data/diagrams/<document_id>/ with a retrievable figure chunk, optionally adds a
VLM description to figures, then reuses this app's existing
embedding config and Qdrant collection (rag.vectorstore.create_or_load_vectorstore)
and document registry (ingest.registry.add_document) so results show up
alongside documents ingested via the existing pipeline.

This module does not replace ingest/embed_and_store.py — call ingest_files_docling()
explicitly when you want the Docling path; the existing ingest_files() is untouched.

Requires: pip install docling  (see requirements.txt)
"""

import base64
import io
import os
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from ingest.registry import add_document
from rag.vectorstore import create_or_load_vectorstore

load_dotenv()

CHUNK_MAX_TOKENS = int(os.getenv("DOCLING_CHUNK_MAX_TOKENS", "700"))
VLM_CAPTION_MODEL = os.getenv("DOCLING_CAPTION_MODEL", "gpt-4o-mini")

# Served by the API at /data/diagrams/... (see the StaticFiles mount in api/main.py).
DIAGRAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "diagrams"

_FIGURE_CAPTION_RE = re.compile(r"^Figure\s+(\d+)\b")
_FIGURE_REF_RE = re.compile(r"\bFigure\s+(\d+)\b")


def _parse_pdf(pdf_path: str):
    """Layout-aware parse: returns Docling's DoclingDocument with structure
    (headings, tables as structured objects, figure regions) intact."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Figure extraction needs the cropped picture images; Docling skips
    # rendering them unless generate_picture_images is set. Scale 2 keeps
    # diagram labels legible when shown in the chat.
    pipeline_options = PdfPipelineOptions(generate_picture_images=True, images_scale=2.0)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(pdf_path)
    return result.document


def _chunk_document(doc, source_path: str) -> list[Document]:
    """
    HybridChunker walks the DoclingDocument tree, merges small runs under
    their heading path, keeps tables as single atomic chunks, and respects
    a max token budget per chunk. Returned as langchain Documents so they
    can flow through the same vectorstore/registry as the existing pipeline.

    Tables are serialized as Markdown tables (the chunker's default is
    "row, column = value" triplets) so the answer can reproduce them as tables.
    Each chunk records the figure numbers its text mentions (`figure_refs`),
    which is how an instruction is linked to its diagram at query time.
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

    # Compact tables: the default pads every cell to the column width, which
    # multiplies the token count of wide tables and makes the chunker shred them.
    params = ChunkingDocSerializer.model_fields["params"].default.model_copy(update={"compact_tables": True})

    class _MarkdownTableProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer(), params=params)

    chunker = HybridChunker(max_tokens=CHUNK_MAX_TOKENS, serializer_provider=_MarkdownTableProvider())

    chunks: list[Document] = []
    seen_texts: set[str] = set()
    for raw_chunk in chunker.chunk(doc):
        meta = raw_chunk.meta
        headings = getattr(meta, "headings", None) or []
        section_path = " > ".join(headings) if headings else ""
        doc_items = getattr(meta, "doc_items", None) or []

        # The table of contents carries no answerable content, and an oversized
        # table can be emitted as repeated identical pieces — keep neither.
        if doc_items and all("document_index" in str(getattr(item, "label", "")).lower() for item in doc_items):
            continue
        if headings and headings[-1].strip().lower() in ("contents", "table of contents"):
            continue
        if raw_chunk.text in seen_texts:
            continue
        seen_texts.add(raw_chunk.text)

        page_no = None
        for item in doc_items:
            prov = getattr(item, "prov", None)
            if prov:
                page_no = prov[0].page_no
                break

        chunk_type = "text"
        for item in doc_items:
            label = str(getattr(item, "label", "")).lower()
            if "table" in label:
                chunk_type = "table"
                break

        chunks.append(Document(
            page_content=raw_chunk.text,
            metadata={
                "source": source_path,
                "page_no": page_no,
                "section_path": section_path,
                "chunk_type": chunk_type,
                "figure_refs": sorted({int(n) for n in _FIGURE_REF_RE.findall(raw_chunk.text)}),
            },
        ))

    return chunks


def _describe_image(client, pil_image) -> str | None:
    """VLM description of a figure, for retrieval beyond what its caption says."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    try:
        resp = client.chat.completions.create(
            model=VLM_CAPTION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe this technical diagram/figure precisely and "
                        "factually, in 2-4 sentences. Name the components, callout "
                        "numbers, and what the figure illustrates (e.g. wiring "
                        "diagram, rating plate, dimension drawing). Do not "
                        "speculate beyond what is visibly labeled."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=250,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [warn] figure description failed: {e}", file=sys.stderr)
        return None


def _extract_figures(doc, source_path: str, document_id: str, describe: bool = False) -> list[Document]:
    """One retrievable chunk per captioned figure, with its image(s) saved to disk.

    Walks the document in reading order. A picture takes Docling's own caption
    when it has one; pictures without one (extra panels of a multi-image
    figure, or figures whose caption Docling labelled as a heading) take the
    next "Figure N ..." line that follows them. Pictures that never get a
    caption (logos, warning icons) are skipped.

    Each chunk's text is the caption (plus a VLM description when `describe`
    is set) and its metadata carries `figure_number`, `figure_caption` and
    `image_paths` (relative to data/, e.g. "diagrams/<document_id>/fig7_1.png").
    """
    from docling_core.types.doc import PictureItem, TextItem

    figures: dict[int, dict] = {}
    pending: list = []  # pictures still waiting for a caption
    section = ""

    def assign(caption: str, pictures: list) -> None:
        match = _FIGURE_CAPTION_RE.match(caption)
        if not match:
            return
        figure = figures.setdefault(int(match.group(1)), {"caption": caption, "section": section, "pictures": []})
        figure["pictures"].extend(pictures)

    for item, _level in doc.iterate_items():
        if isinstance(item, PictureItem):
            caption = item.caption_text(doc).strip()
            if caption:
                assign(caption, pending + [item])
                pending = []
            else:
                pending.append(item)
        elif isinstance(item, TextItem):
            text = item.text.strip()
            if _FIGURE_CAPTION_RE.match(text):
                if pending:
                    assign(text, pending)
                    pending = []
            elif "section_header" in str(item.label):
                section = text

    client = None
    if describe:
        from openai import OpenAI
        client = OpenAI()

    dest_dir = DIAGRAMS_DIR / document_id
    chunks: list[Document] = []
    for number in sorted(figures):
        figure = figures[number]
        image_paths: list[str] = []
        descriptions: list[str] = []
        page_no = None
        for picture in figure["pictures"]:
            try:
                pil_image = picture.get_image(doc)
            except Exception as e:
                print(f"  [warn] could not extract image for Figure {number}: {e}", file=sys.stderr)
                continue
            if pil_image is None:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            name = f"fig{number}_{len(image_paths) + 1}.png"
            pil_image.save(dest_dir / name, format="PNG")
            image_paths.append(f"diagrams/{document_id}/{name}")
            if page_no is None and picture.prov:
                page_no = picture.prov[0].page_no
            if client is not None:
                description = _describe_image(client, pil_image)
                if description:
                    descriptions.append(description)
        if not image_paths:
            continue

        chunks.append(Document(
            page_content="\n".join([figure["caption"], *descriptions]),
            metadata={
                "source": source_path,
                "page_no": page_no,
                "section_path": figure["section"],
                "chunk_type": "figure",
                "figure_number": number,
                "figure_caption": figure["caption"],
                "image_paths": image_paths,
            },
        ))

    return chunks


def _attach_metadata(chunks: list[Document], document_id: str, document_name: str, version: str, metadata: dict):
    for chunk in chunks:
        chunk.metadata.update({
            "document_id": document_id,
            "document_name": document_name,
            "version": version,
            **metadata,
        })
    return chunks


def ingest_files_docling(
    file_paths: list[str],
    metadata: dict[str, str] | None = None,
    version: str = "1.0",
    original_names: list[str] | None = None,
    caption_figures: bool = False,
):
    """Docling-based alternative to ingest.embed_and_store.ingest_files.

    Parses each PDF with Docling (layout + table structure aware), chunks
    with HybridChunker (header-aware, table-atomic, Markdown tables), saves
    captioned figures as images with a figure chunk each, and stores in the
    same Qdrant collection / document registry as the existing ingestion pipeline.

    Only PDF files are supported (Docling's layout/table analysis targets
    PDFs); pass files of other types through ingest.embed_and_store.ingest_files
    instead.

    Args:
        file_paths: Paths to PDF documents to ingest.
        metadata: Arbitrary key-value tags attached to every chunk of every
            document (e.g. department, classification) — caller-defined.
        version: Document version string.
        original_names: Human-readable file names to record as document_name,
            parallel to file_paths. Defaults to each path's own file name —
            pass this when file_paths point at temp files with generated names.
        caption_figures: If True, also run each figure through a VLM and add
            its description to the figure chunk's text (the document's own
            caption is always used).
    """
    metadata = metadata or {}
    if original_names is None:
        original_names = [Path(p).name for p in file_paths]

    for path in file_paths:
        if Path(path).suffix.lower() != ".pdf":
            raise ValueError(f"Docling ingestion only supports PDF files, got: {path}")

    all_chunks: list[Document] = []
    document_records = []

    for path, doc_name in zip(file_paths, original_names):
        print(f"Parsing {doc_name} with Docling...")
        doc = _parse_pdf(path)

        print("Chunking (header-aware, table-atomic)...")
        path_chunks = _chunk_document(doc, source_path=path)
        print(f"  -> {len(path_chunks)} text/table chunks")

        doc_id = str(uuid.uuid4())
        print("Extracting figures" + (" (with VLM descriptions)..." if caption_figures else "..."))
        fig_chunks = _extract_figures(doc, source_path=path, document_id=doc_id, describe=caption_figures)
        print(f"  -> {len(fig_chunks)} figure chunks")
        path_chunks.extend(fig_chunks)

        _attach_metadata(path_chunks, document_id=doc_id, document_name=doc_name, version=version, metadata=metadata)
        document_records.append({"document_id": doc_id, "document_name": doc_name, "chunk_count": len(path_chunks)})
        all_chunks.extend(path_chunks)

    if not all_chunks:
        print("⚠️ No chunks produced.")
        return None

    print(f"Uploading {len(all_chunks)} chunks to Qdrant...")
    create_or_load_vectorstore(documents=all_chunks)
    print("✅ Docling ingestion successfully completed.")

    for record in document_records:
        add_document(
            document_id=record["document_id"],
            document_name=record["document_name"],
            version=version,
            metadata=metadata,
            chunk_count=record["chunk_count"],
        )

    return {"total_chunks": len(all_chunks), "documents": document_records}


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.embed_and_store_docling <file1.pdf> [file2.pdf ...]")
        print("  Env vars: VERSION, METADATA (JSON object string), CAPTION_FIGURES (1/0)")
        return

    import json
    paths = [str(Path(p)) for p in sys.argv[1:]]
    try:
        result = ingest_files_docling(
            paths,
            metadata=json.loads(os.getenv("METADATA", "{}")),
            version=os.getenv("VERSION", "1.0"),
            caption_figures=os.getenv("CAPTION_FIGURES", "0") == "1",
        )
        if result:
            print(f"Total chunks indexed: {result['total_chunks']}")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")


if __name__ == "__main__":
    main()
