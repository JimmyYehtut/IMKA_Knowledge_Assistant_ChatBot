
from pathlib import Path
import os
import sys
import uuid

from pip._vendor import truststore as t
t.extract_from_ssl()

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")
sys.path.insert(0, str(BACKEND_DIR))

from rag.qdrant_init import ensure_collection, get_qdrant_client
from ingest.embed_and_store_docling import (
    _parse_pdf,
    _chunk_document,
)

COLLECTION_NAME = "imka_vlm_docling_test"
EXPECTED_COLLECTION = "imka_vlm_docling_test"

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

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))


def main():
    # Safety checks: never write to the production collection.
    if COLLECTION_NAME != EXPECTED_COLLECTION:
        raise RuntimeError("Unexpected collection name. Stopping.")

    if COLLECTION_NAME == "ragindex":
        raise RuntimeError("Production collection is forbidden for this test.")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"ABB manual not found: {PDF_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Figure 38 image not found: {IMAGE_PATH}")

    client = get_qdrant_client()

    # Do not overwrite or add duplicates to an existing test collection.
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count > 0:
            raise RuntimeError(
                f"{COLLECTION_NAME} already contains {info.points_count} "
                "points. Nothing was written."
            )
        print(f"Empty test collection already exists: {COLLECTION_NAME}")
    else:
        ensure_collection(
            client, COLLECTION_NAME, EMBEDDING_DIMENSION
        )
        print(f"Created isolated collection: {COLLECTION_NAME}")

    print("Parsing ABB manual to extract Figure 38 context...")
    doc = _parse_pdf(str(PDF_PATH))
    text_chunks = _chunk_document(
        doc, source_path=PDF_PATH.name
    )

    figure_number = 38
    figure_page = 91

    relevant_chunks = [
        chunk
        for chunk in text_chunks
        if (
            chunk.metadata.get("page_no") == figure_page
            and (
                figure_number in chunk.metadata.get("figure_refs", [])
                or "Figure 38" in chunk.metadata.get("section_path", "")
            )
        )
    ]

    if not relevant_chunks:
        raise RuntimeError(
            "No context chunks found for Figure 38 on printed page 91."
        )

    figure_context = "\n\n".join(
        chunk.page_content for chunk in relevant_chunks
    )

    # Use the existing VLM-generated description from the successful test.
    # This avoids another vision-model API call.
    figure_description = """
    Figure 38: Correlation between the insulation resistance and the temperature.
    ABB Manual for Induction Motors and Generators, printed page 91, section 7.6.4.1.

    The vertical axis (1) represents the coefficient Kt for insulation resistance
    on a logarithmic scale. The horizontal axis (2) represents winding temperature
    in degrees Celsius. The graph is used to convert a measured insulation
    resistance R at a particular temperature to an equivalent value R40 at 40°C.

    Formula: R40 = k x R.
    The manual's printed example is R = 30 MΩ measured at 20°C, k = 0.25,
    giving R40 = 0.25 x 30 MΩ = 7.5 MΩ.

    The manual cautions that the graph should be used for temperatures fairly
    near 40°C, because large deviations may cause errors. Distinguish the
    printed example from any approximate coefficient read from the graph.
    """

    # Store the description and the supporting extracted manual context.
    documents = [
        Document(
            page_content=figure_description,
            metadata={
                "source": PDF_PATH.name,
                "page_no": figure_page,
                "figure": "Figure 38",
                "chunk_type": "vlm_figure_description",
                "test_collection": True,
            },
        ),
        Document(
            page_content=figure_context,
            metadata={
                "source": PDF_PATH.name,
                "page_no": figure_page,
                "figure_refs": [figure_number],
                "chunk_type": "text",
                "test_collection": True,
            },
        ),
    ]

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    print(f"Writing {len(documents)} test documents...")
    vectorstore.add_documents(documents)

    print(
        "Test collection point count:",
        client.get_collection(COLLECTION_NAME).points_count,
    )

    query = (
        "What is the formula for converting measured insulation "
        "resistance to its equivalent value at 40°C, and what is "
        "the printed worked example in Figure 38?"
    )

    results = vectorstore.similarity_search_with_score(query, k=2)

    print("\n--- Retrieval test ---")
    if not results:
        raise RuntimeError("No documents were retrieved.")

    for retrieved_doc, score in results:
        print(f"\nSimilarity score: {score}")
        print("Metadata:", retrieved_doc.metadata)
        print(retrieved_doc.page_content[:1200])

    print("\nPASS: isolated Figure 38 storage and retrieval completed.")
    print("The document registry was not modified.")
    print("The production collection ragindex was not targeted.")


if __name__ == "__main__":
    main()
