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
# gpt-5.6-luna reads chart curves and plate values correctly where gpt-4o / gpt-4o-mini
# invented axis titles and misread a log-scale curve (ABB Figure 38). "qwen" uses the
# Qwen3.8-27B RunPod endpoint (rag/qwen_client.py) instead of OpenAI.
VLM_CAPTION_MODEL = os.getenv("DOCLING_CAPTION_MODEL", "gpt-5.6-luna")
CAPTION_CONCURRENCY = int(os.getenv("DOCLING_CAPTION_CONCURRENCY", "8"))
# Defaults for ingest_files_docling(), so .env drives both the CLI and POST /api/documents.
CAPTION_FIGURES = os.getenv("CAPTION_FIGURES", "0") == "1"

# Served by the API at /data/diagrams/... (see the StaticFiles mount in api/main.py).
DIAGRAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "diagrams"

# Figure labels as the manuals print them: "Figure 12" (ABB, Siemens), "Fig. 27:" (Kohler),
# "Figure-16a" (EMSA), "FIGURE 25–1" (Keljik, chapter–figure). The number part becomes the
# figure's id via _figure_id ("12", "27", "16a", "25-1").
_FIGURE_NUMBER = r"(?:Figure|Fig\.)[\s\-–]*(\d+(?:\s*[–\-]\s*\d+)?[a-z]?)\b"
_FIGURE_CAPTION_RE = re.compile(r"^" + _FIGURE_NUMBER, re.IGNORECASE)
_FIGURE_REF_RE = re.compile(r"\b" + _FIGURE_NUMBER, re.IGNORECASE)


def _figure_id(number: str) -> str:
    """'25 – 1' -> '25-1', '16A' -> '16a': one spelling per figure, for captions and references alike."""
    return re.sub(r"\s+", "", number).replace("–", "-").lower()


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
                "figure_refs": sorted({_figure_id(n) for n in _FIGURE_REF_RE.findall(raw_chunk.text)}),
            },
        ))

    return chunks


_DESCRIBE_PROMPT = """\
This image is a figure from a technical equipment manual{caption}.{legend}
Write a factual description that lets someone answer questions about its content
without seeing it. Start with one sentence saying what the figure shows, then give
the content that applies:

- Chart or graph: the axis titles, units and ranges, whether a scale is logarithmic,
  each curve or series and its label, and readings taken from the curves as
  "x = ... -> y = ..." pairs (at least 4 per curve, spread over the range; prefix a
  value you have to estimate with "≈").
- Table, rating plate or nameplate: transcribe every field and value exactly.
- Drawing or diagram: every callout number with what it points to, and all
  dimensions, angles, clearances and other numbers with their units.
- Any other text visible in the image, verbatim.

Report only what is visible. Do not guess unreadable values and do not add
information that is not in the image."""


def caption_model_name() -> str:
    """The figure-description model, as recorded in the registry."""
    if VLM_CAPTION_MODEL == "qwen":
        from rag.qwen_client import model_alias
        return model_alias()
    return VLM_CAPTION_MODEL


def _describe_image(client, pil_image, caption: str | None = None, legend: list[str] | None = None) -> str | None:
    """VLM description of a figure, including its values, for retrieval and answering.

    `legend` is the text printed between the figure and its caption; the model uses it to
    name callout numbers and unlabeled axes ("1" = "Coefficient Kt for Insulation Resistance").
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    legend_text = ""
    if legend:
        lines = "; ".join(f'"{line}"' for line in legend)
        legend_text = (f"\nThe legend printed under the figure reads, in order: {lines}. Use it to name the "
                       "numbered callouts and unlabeled axes in the image (the first entry is usually 1).")
    prompt = _DESCRIBE_PROMPT.format(caption=f', captioned "{caption}"' if caption else "", legend=legend_text)
    if VLM_CAPTION_MODEL == "qwen":
        from rag.qwen_client import describe_image
        try:
            # Low temperature for transcription (endpoint README: 0.2-0.3 for OCR/extraction).
            # Thinking is off (qwen_client), so the whole budget is description; Qwen's ABB
            # descriptions ran to ~270 tokens (max ~960), so 4000 is headroom, as for luna.
            return describe_image(prompt, f"data:image/png;base64,{b64}", temperature=0.2, max_tokens=4000)
        except Exception as e:
            print(f"  [warn] figure description failed: {e}", file=sys.stderr)
            return None
    # Reasoning models (gpt-5.x, o-series) take max_completion_tokens, which also covers
    # their reasoning tokens. Every captioner gets 4000, as Qwen does, so none is cut short
    # where another is not (descriptions ran to ~270 tokens, the longest ~960).
    reasoning = VLM_CAPTION_MODEL.startswith(("gpt-5", "o1", "o3", "o4"))
    # Non-reasoning models transcribe at 0.2, as Qwen does; reasoning models accept only
    # their default temperature.
    limit = {"max_completion_tokens": 4000} if reasoning else {"max_tokens": 4000, "temperature": 0.2}
    try:
        resp = client.chat.completions.create(
            model=VLM_CAPTION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            **limit,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [warn] figure description failed: {e}", file=sys.stderr)
        return None


# A "Figure N ..." line Docling labelled as plain text and longer than this is body
# text that mentions a figure ("Figure 14 Main terminal box is an example of how ..."),
# not its caption. Lines labelled caption / section_header count at any length.
_MAX_CAPTION_WORDS = 16
# Layout slack (points) when comparing a caption's top with a picture's bottom.
_CAPTION_GAP_SLACK = 6.0
# A picture under a figure caption that is this much smaller than the figure's largest
# picture is a stray icon or logo, not a panel of the figure.
_MIN_PANEL_AREA_RATIO = 0.1


def _bottom_left(item, doc):
    """(page_no, bbox in bottom-left page coordinates) of an item's first provenance, or None."""
    if not item.prov:
        return None
    prov = item.prov[0]
    page_height = doc.pages[prov.page_no].size.height
    return prov.page_no, prov.bbox.to_bottom_left_origin(page_height)


def _overlaps_horizontally(a, b) -> bool:
    return min(a.r, b.r) > max(a.l, b.l)


def _group_figures(doc) -> dict[str, dict]:
    """{figure id: {"caption", "section", "pictures"}} for every captioned picture.

    Docling's own caption links cannot be trusted alone: it attaches a caption to
    the wrong picture when two figures are stacked on a page, labels some captions
    as headings, and leaves extra panels of a figure uncaptioned. Following reading
    order instead carries uncaptioned pictures (cover photo, warning icons) forward
    to whatever caption comes next, pages later. So pictures are matched by layout:

    - caption candidates are figure-label lines ("Figure N", "Fig. N:", "Figure-Na",
      "FIGURE N–M") labelled caption or section header,
      plus short ones labelled plain text;
    - a picture takes the nearest candidate *below* it on the same page, preferring
      one that overlaps it horizontally (two-column layouts). Stacked figures each
      reach their own caption first; panels of a grid figure all reach its caption;
    - only when the page has no such caption does Docling's own "Figure N" caption
      on the picture count. Anything else stays uncaptioned and is skipped;
    - a picture far smaller than the largest one under the same caption is dropped
      as a stray icon.
    """
    from docling_core.types.doc import PictureItem, TextItem

    captions: list[dict] = []
    pictures: list[tuple] = []  # (item, page_no, bbox, section at the picture)
    texts: list[tuple] = []  # (text, page_no, bbox) of every other text line — legend candidates
    section = ""
    for item, _level in doc.iterate_items():
        if isinstance(item, PictureItem):
            where = _bottom_left(item, doc)
            if where:
                pictures.append((item, *where, section))
        elif isinstance(item, TextItem):
            text = item.text.strip()
            match = _FIGURE_CAPTION_RE.match(text)
            label = str(item.label)
            where = _bottom_left(item, doc)
            if match and ("caption" in label or "section_header" in label
                          or len(text.split()) <= _MAX_CAPTION_WORDS):
                if where:
                    captions.append({"number": _figure_id(match.group(1)), "text": text, "section": section,
                                     "page": where[0], "bbox": where[1]})
            else:
                if not match and "section_header" in label:
                    section = text
                if where and text:
                    texts.append((text, *where))

    figures: dict[str, dict] = {}
    for picture, page_no, bbox, picture_section in pictures:
        below = [c for c in captions if c["page"] == page_no and c["bbox"].t <= bbox.b + _CAPTION_GAP_SLACK]
        overlapping = [c for c in below if _overlaps_horizontally(c["bbox"], bbox)]
        candidates = overlapping or below
        caption = min(candidates, key=lambda c: bbox.b - c["bbox"].t) if candidates else None
        if caption is None:
            own_text = picture.caption_text(doc).strip()
            match = _FIGURE_CAPTION_RE.match(own_text)
            if match:
                caption = {"number": _figure_id(match.group(1)), "text": own_text, "section": picture_section}
        if caption is None:
            continue
        figure = figures.setdefault(caption["number"], {"caption": caption["text"], "section": caption["section"],
                                                        "pictures": [], "areas": [], "legend": []})
        figure["pictures"].append(picture)
        figure["areas"].append(bbox.width * bbox.height)
        # Legend: text lines between this picture's bottom and its caption ("1 Coefficient Kt for
        # Insulation Resistance") — what the picture's bare callout numbers and axis labels mean.
        if "bbox" in caption:
            for text, page, text_bbox in texts:
                if (page == page_no and text_bbox.t <= bbox.b + _CAPTION_GAP_SLACK
                        and text_bbox.b >= caption["bbox"].t - _CAPTION_GAP_SLACK
                        and (_overlaps_horizontally(text_bbox, bbox)
                             or _overlaps_horizontally(text_bbox, caption["bbox"]))
                        and text not in figure["legend"]):
                    figure["legend"].append(text)

    for figure in figures.values():
        largest = max(figure["areas"])
        figure["pictures"] = [p for p, area in zip(figure["pictures"], figure.pop("areas"))
                              if area >= _MIN_PANEL_AREA_RATIO * largest]
    return figures


def _extract_figures(doc, source_path: str, document_id: str, describe: bool = False) -> list[Document]:
    """One retrievable chunk per captioned figure, with its image(s) saved to disk.

    Pictures are grouped by figure with _group_figures (caption below the
    picture on the same page). Pictures that get no "Figure N" caption (logos,
    cover photos, warning icons) are skipped.

    Each chunk's text is the caption (plus a VLM description when `describe`
    is set) and its metadata carries `figure_number`, `figure_caption` and
    `image_paths` (relative to data/, e.g. "diagrams/<document_id>/fig7_1.png").
    """
    figures = _group_figures(doc)

    client = None
    if describe:
        if VLM_CAPTION_MODEL == "qwen":
            client = "qwen"  # _describe_image calls rag.qwen_client itself
        else:
            from openai import OpenAI
            client = OpenAI()

    dest_dir = DIAGRAMS_DIR / document_id
    saved: list[tuple[str, list[str], list, int | None]] = []  # (number, image paths, images, page)
    for number in sorted(figures):
        figure = figures[number]
        image_paths: list[str] = []
        images = []
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
            images.append(pil_image)
            if page_no is None and picture.prov:
                page_no = picture.prov[0].page_no
        if image_paths:
            saved.append((number, image_paths, images, page_no))

    # VLM descriptions are independent calls (a reasoning model takes ~30 s each), so run them
    # concurrently; results keep the figure and panel order.
    descriptions: dict[tuple[str, int], str | None] = {}
    if client is not None:
        from concurrent.futures import ThreadPoolExecutor

        jobs = [(number, i, image) for number, _, images, _ in saved for i, image in enumerate(images)]
        with ThreadPoolExecutor(max_workers=CAPTION_CONCURRENCY) as pool:
            results = pool.map(
                lambda job: _describe_image(client, job[2], figures[job[0]]["caption"], figures[job[0]]["legend"]),
                jobs,
            )
            descriptions = {(number, i): text for (number, i, _), text in zip(jobs, results)}

    chunks: list[Document] = []
    for number, image_paths, images, page_no in saved:
        figure = figures[number]
        figure_descriptions = [d for i in range(len(images)) if (d := descriptions.get((number, i)))]
        chunks.append(Document(
            # The legend goes into the text either way: it is often the only place the figure's
            # quantities are named ("Coefficient Kt"), so retrieval needs it even without a VLM.
            page_content="\n".join([figure["caption"],
                                    *([f"Legend: {'; '.join(figure['legend'])}"] if figure["legend"] else []),
                                    *figure_descriptions]),
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
    caption_figures: bool = CAPTION_FIGURES,
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
            pipeline="docling",
            chunking=True,
            captioning=caption_figures,
            models={"captioning": caption_model_name()} if caption_figures else {},
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
        )
        if result:
            print(f"Total chunks indexed: {result['total_chunks']}")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")


if __name__ == "__main__":
    main()
