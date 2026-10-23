
from pip._vendor import truststore as t
t.extract_from_ssl()

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from rag.qdrant_init import ensure_collection, get_qdrant_client

# Load the existing backend configuration.
load_dotenv(Path(__file__).resolve().parent / ".env")

COLLECTION_NAME = "imka_vlm_test"
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

client = get_qdrant_client()
ensure_collection(client, COLLECTION_NAME, EMBEDDING_DIMENSION)

# Safety check: do not add duplicate test points.
info = client.get_collection(COLLECTION_NAME)
if info.points_count != 0:
    raise RuntimeError(
        f"{COLLECTION_NAME} already contains {info.points_count} points. "
        "Nothing was added. Stop and inspect the collection first."
    )

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIMENSION,
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

figure_description = """
Document: ABB Manual for Induction Motors and Generators.
Printed page: 91.
Section 7.6.4.1: Conversion of measured insulation resistance values.
Figure 38 title: Correlation between the insulation resistance and the temperature.

Purpose:
Figure 38 is used to convert measured insulation resistance R at a particular
winding temperature into an equivalent insulation resistance R40 at 40 degrees
Celsius.

Formula printed in the manual:
R40 = k x R

Axes:
The horizontal x-axis is winding temperature in degrees Celsius, from 0 to
110 degrees Celsius, on a linear scale.
The vertical y-axis is the coefficient Kt for insulation resistance, on a
logarithmic scale.

Approximate coefficients read from the plotted curve:
At 20 degrees Celsius, k is approximately 0.25.
At 30 degrees Celsius, k is approximately 0.5.
At 40 degrees Celsius, k is approximately 1.0.
These are approximate graph readings, not exact tabulated values.

Worked example printed in the manual:
R = 30 megaohms measured at 20 degrees Celsius.
k = 0.25.
R40 = 0.25 x 30 megaohms = 7.5 megaohms.

Example calculation using the approximate graph reading at 30 degrees Celsius:
If measured insulation resistance is 50 megaohms at 30 degrees Celsius,
using k approximately 0.5 gives R40 approximately 0.5 x 50 = 25 megaohms.
This is an approximate graph-based calculation, not the printed worked example.

The manual cautions that this graph should be used for temperatures fairly
near the standard value of 40 degrees Celsius, because large temperature
deviations can cause errors.
"""

document = Document(
    page_content=figure_description,
    metadata={
        "source": "ABB Manual for Induction Motors and Generators_EN.pdf",
        "page": 91,
        "figure": "Figure 38",
        "content_type": "vlm_figure_description",
        "test_collection": True,
    },
)

vectorstore.add_documents([document])

print(f"Added one Figure 38 description to {COLLECTION_NAME}.")
print(f"Point count: {client.get_collection(COLLECTION_NAME).points_count}")

queries = [
    "What correction coefficient is used for insulation resistance measured at 30 degrees Celsius?",
    "How do I convert 50 megaohms measured at 30 degrees Celsius to the equivalent resistance at 40 degrees Celsius?",
    "What formula does ABB use to convert measured insulation resistance to R40?",
]

for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")

    results = vectorstore.similarity_search_with_score(query, k=1)

    if not results:
        print("No matching document retrieved.")
        continue

    retrieved_doc, score = results[0]
    print(f"Similarity score: {score}")
    print("Retrieved figure description:")
    print(retrieved_doc.page_content[:1200])
