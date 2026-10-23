
from pathlib import Path

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
    text_chunks = _chunk_document(
        doc,
        source_path=PDF_PATH.name,
    )

    figure_number = 38
    figure_page = 91

    relevant_chunks = [
        chunk
        for chunk in text_chunks
        if (
            chunk.metadata.get("page_no") == figure_page
            and (
                figure_number
                in chunk.metadata.get("figure_refs", [])
                or (
                    "Figure 38"
                    in chunk.metadata.get("section_path", "")
                )
            )
        )
    ]

    figure_context = "\n\n".join(
        chunk.page_content for chunk in relevant_chunks
    )

    print(f"\nSelected chunks: {len(relevant_chunks)}")

    for i, chunk in enumerate(relevant_chunks, start=1):
        print(f"\n--- Selected chunk {i} ---")
        print("Metadata:", chunk.metadata)
        print(chunk.page_content)

    print("\n--- Combined Figure 38 Context ---")
    print(figure_context)

    assert len(relevant_chunks) == 2, (
        f"Expected 2 chunks, found {len(relevant_chunks)}"
    )
    assert "R40 = k x R" in figure_context
    assert "7.5 M" in figure_context

    print("\nPASS: Figure 38 context selection is correct.")


if __name__ == "__main__":
    main()
