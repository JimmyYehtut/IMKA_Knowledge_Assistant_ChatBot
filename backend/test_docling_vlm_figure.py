
import sys
import uuid
from pathlib import Path

from pip._vendor import truststore as t
t.extract_from_ssl()

from dotenv import load_dotenv

# Load the existing backend environment configuration.
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

# Import only the parsing and extraction functions.
# Do not call ingest_files_docling(), because it writes to Qdrant
# and updates the main document registry.
from ingest.embed_and_store_docling import (
    _parse_pdf,
    _chunk_document,
    _extract_figures,
)

PDF_PATH = BACKEND_DIR / "data" / (
    "ABB Manual for Induction Motors and Generators_EN.pdf"
)

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"ABB manual not found: {PDF_PATH}")

    print(f"Parsing PDF: {PDF_PATH.name}")
    doc = _parse_pdf(str(PDF_PATH))

    
    print("Extracting text chunks for figure context...")
    text_chunks = _chunk_document(
        doc,
        source_path=PDF_PATH.name,
    )

    print("Extracting figures with contextual VLM descriptions enabled...")
    
    figures = _extract_figures(
        doc,
        source_path=PDF_PATH.name,
        document_id=f"test-{uuid.uuid4()}",
        describe=True,
        text_chunks=text_chunks,
        figure_numbers={38},
    )



    print(f"\nTotal extracted figure chunks: {len(figures)}")

    # Inspect Figure 38 only; do not upload any chunks.
    matches = [
        figure for figure in figures
        if figure.metadata.get("figure_number") == 38
    ]

    if not matches:
        print("Figure 38 was not found as a captioned figure.")
        print("No Qdrant or registry writes were performed.")
        return

    for figure in matches:
        print("\n" + "=" * 70)
        print("FIGURE 38 — METADATA")
        print("=" * 70)
        print(figure.metadata)

        print("\nFIGURE 38 — CAPTION AND VLM DESCRIPTION")
        print("=" * 70)
        print(figure.page_content)

    print("\nCompleted extraction-only test.")
    print("No Qdrant uploads or registry updates were performed.")

if __name__ == "__main__":
    main()
