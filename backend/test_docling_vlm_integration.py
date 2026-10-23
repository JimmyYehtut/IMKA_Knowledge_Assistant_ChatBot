
from pathlib import Path

from pip._vendor import truststore as t
t.extract_from_ssl()

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

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
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print("Parsing ABB manual with Docling...")
    doc = _parse_pdf(str(PDF_PATH))

    print("\nExtracting text chunks...")
    text_chunks = _chunk_document(
        doc,
        source_path=PDF_PATH.name,
    )

    print("\nExtracting figures with VLM descriptions...")
    figure_chunks = _extract_figures(
        doc,
        source_path=PDF_PATH.name,
        document_id="integration-test",
        describe=True,
    )

    # Select relevant text using the actual section and content.
    relevant_text = [
        chunk for chunk in text_chunks
        if (
            "7.6.4.1" in chunk.metadata.get("section_path", "")
            or "R40" in chunk.page_content
            or "Kt for Insulation Resistance" in chunk.page_content
        )
    ]

    figure_38 = [
        chunk for chunk in figure_chunks
        if chunk.metadata.get("figure_number") == 38
    ]

    print(f"\nRelevant text chunks: {len(relevant_text)}")
    for index, chunk in enumerate(relevant_text, start=1):
        print("\n" + "=" * 70)
        print(f"TEXT CHUNK {index}")
        print("Metadata:", chunk.metadata)
        print(chunk.page_content)

    print(f"\nFigure 38 chunks found: {len(figure_38)}")
    for chunk in figure_38:
        print("\n" + "=" * 70)
        print("FIGURE 38 METADATA")
        print(chunk.metadata)
        print("\nFIGURE 38 CONTENT")
        print(chunk.page_content)

    print("\nExtraction-only integration test completed.")
    print("No Qdrant uploads or registry updates were performed.")

if __name__ == "__main__":
    main()
