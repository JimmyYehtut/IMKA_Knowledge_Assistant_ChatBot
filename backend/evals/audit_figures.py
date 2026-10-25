"""Dry-run the Docling figure extraction on a PDF and compare it with what is indexed.

    cd backend
    python -m evals.audit_figures "../Dataset/manual.pdf"                 # parse + extract
    python -m evals.audit_figures "../Dataset/manual.pdf" --save-doc parsed.json
    python -m evals.audit_figures parsed.json --document-id <id>          # reuse a parse, diff with Qdrant

Nothing is written to Qdrant, the registry or backend/data: figure images go to
a temporary folder that is printed at the end, so they can be opened and
checked by eye. Parsing a 150-page manual takes a few minutes on CPU; save the
parse with --save-doc and pass the JSON next time to skip it.

With --document-id, each figure is compared with the figure chunk stored in
Qdrant for that document (page and image count), which shows whether the index
needs re-ingesting after an extraction fix.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def load_document(path: Path, save_doc: Path | None):
    from docling_core.types.doc import DoclingDocument, ImageRefMode

    if path.suffix.lower() == ".json":
        return DoclingDocument.load_from_json(path)
    from ingest.embed_and_store_docling import _parse_pdf

    print(f"Parsing {path.name} with Docling (a few minutes)...", flush=True)
    doc = _parse_pdf(str(path))
    if save_doc:
        doc.save_as_json(save_doc, image_mode=ImageRefMode.EMBEDDED)
        print(f"Saved parse to {save_doc}")
    return doc


def indexed_figures(document_id: str) -> dict[int, dict]:
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    from rag.qdrant_init import get_qdrant_client
    from rag.vectorstore import QDRANT_COLLECTION_NAME

    points, _ = get_qdrant_client().scroll(
        collection_name=QDRANT_COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="metadata.chunk_type", match=MatchValue(value="figure")),
            FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id)),
        ]),
        limit=1000, with_payload=True, with_vectors=False,
    )
    return {p.payload["metadata"]["figure_number"]: p.payload["metadata"] for p in points}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="PDF to parse, or a DoclingDocument JSON saved with --save-doc")
    parser.add_argument("--save-doc", type=Path, help="save the Docling parse here for reuse")
    parser.add_argument("--document-id", help="compare with the figure chunks indexed for this document")
    args = parser.parse_args()

    import ingest.embed_and_store_docling as docling_ingest

    doc = load_document(args.source, args.save_doc)
    out_dir = Path(tempfile.mkdtemp(prefix="figure_audit_"))
    docling_ingest.DIAGRAMS_DIR = out_dir  # keep backend/data untouched
    chunks = docling_ingest._extract_figures(doc, source_path=str(args.source), document_id="audit")
    extracted = {c.metadata["figure_number"]: c.metadata for c in chunks}
    stored = indexed_figures(args.document_id) if args.document_id else {}

    print(f"\n{len(extracted)} figures extracted" + (f", {len(stored)} indexed" if args.document_id else ""))
    print(f"{'Fig':>4}  {'page':>5}  {'imgs':>4}  " + ("indexed page/imgs  " if args.document_id else "") + "caption")
    differences = 0
    for number in sorted(set(extracted) | set(stored)):
        new, old = extracted.get(number), stored.get(number)
        page = new["page_no"] if new else "-"
        imgs = len(new["image_paths"]) if new else "-"
        caption = (new or old)["figure_caption"][:70]
        line = f"{number:>4}  {page!s:>5}  {imgs!s:>4}  "
        if args.document_id:
            was = f"{old['page_no']}/{len(old['image_paths'])}" if old else "missing"
            differs = not old or not new or (old["page_no"], len(old["image_paths"])) != (page, imgs)
            differences += differs
            line += f"{was:>8}{'  <-- differs' if differs else '':<13}"
        print(line + caption)

    if args.document_id:
        print(f"\n{differences} figure(s) differ from the index." + (" Re-ingest to apply." if differences else ""))
    print(f"Extracted images: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
