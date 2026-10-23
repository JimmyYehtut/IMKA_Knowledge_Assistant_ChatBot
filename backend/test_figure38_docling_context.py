
from pathlib import Path

from pip._vendor import truststore as t
t.extract_from_ssl()

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

import sys
sys.path.insert(0, str(BACKEND_DIR))

from ingest.embed_and_store_docling import (
    _parse_pdf,
    _chunk_document,
    _describe_image,
)

PDF_PATH = BACKEND_DIR / "data" / (
    "ABB Manual for Induction Motors and Generators_EN.pdf"
)

IMAGE_PATH = (
    BACKEND_DIR
    / "data"
    / "diagrams"
    / "integration-test"
    / "fig38_1.png"
)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Figure 38 image not found: {IMAGE_PATH}")

    # Step 1: Extract actual text chunks from the ABB manual.
    print("Parsing ABB manual...")
    doc = _parse_pdf(str(PDF_PATH))

    print("Creating text chunks...")
    text_chunks = _chunk_document(
        doc,
        source_path=PDF_PATH.name,
    )

    # Step 2: Select context for Figure 38 on page 91.
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

    if not relevant_chunks:
        raise RuntimeError("No relevant text chunks found for Figure 38.")

    figure_context = "\n\n".join(
        chunk.page_content for chunk in relevant_chunks
    )

    print(f"Selected {len(relevant_chunks)} context chunks.")

    # Step 3: Call the VLM once using the saved image and actual context.
    print("Calling the vision-language model...")
    with Image.open(IMAGE_PATH) as image:
        image.load()
        client = OpenAI()
        description = _describe_image(
            client,
            image,
            context=figure_context,
        )

    print("\n--- Automatically Contextualized Figure 38 Description ---")
    print(description or "No description was returned.")


if __name__ == "__main__":
    main()
