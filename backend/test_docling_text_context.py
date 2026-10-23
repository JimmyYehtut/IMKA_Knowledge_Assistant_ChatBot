
from pathlib import Path

from pip._vendor import truststore as t
t.extract_from_ssl()

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

from ingest.embed_and_store_docling import _parse_pdf, _chunk_document

PDF_PATH = BACKEND_DIR / "data" / (
    "ABB Manual for Induction Motors and Generators_EN.pdf"
)

def main():
    print(f"Parsing: {PDF_PATH.name}")
    doc = _parse_pdf(str(PDF_PATH))

    print("Creating text chunks...")
    chunks = _chunk_document(
        doc,
        source_path=PDF_PATH.name,
    )

    matches = []
    for chunk in chunks:
        text = chunk.page_content
        metadata = chunk.metadata
        section = metadata.get("section_path", "")

        if (
            "7.6.4.1" in section
            or "insulation resistance" in text.lower()
            or "r40" in text.lower()
        ):
            matches.append(chunk)

    print(f"\nMatching text chunks: {len(matches)}")

    for i, chunk in enumerate(matches, start=1):
        print("\n" + "=" * 70)
        print(f"CHUNK {i}")
        print("Metadata:", chunk.metadata)
        print("-" * 70)
        print(chunk.page_content[:2500])

if __name__ == "__main__":
    main()
