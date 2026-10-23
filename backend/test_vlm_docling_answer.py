
from pip._vendor import truststore as t
t.extract_from_ssl()

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

import sys
sys.path.insert(0, str(BACKEND_DIR))

from rag.qdrant_init import get_qdrant_client

# Safety: read only from the isolated test collection.
COLLECTION_NAME = "imka_vlm_docling_test"

if COLLECTION_NAME == "ragindex":
    raise RuntimeError("Production collection is forbidden for this test.")

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

client = get_qdrant_client()

if not client.collection_exists(COLLECTION_NAME):
    raise RuntimeError(
        f"Test collection {COLLECTION_NAME} does not exist. Stopping."
    )

info = client.get_collection(COLLECTION_NAME)
if info.points_count == 0:
    raise RuntimeError("Test collection is empty. Stopping.")

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIMENSION,
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = (
    "Answer using the retrieved ABB manual context and any explicit "
    "test-specific graph-reading guidance supplied in the question. "
    "Distinguish exact printed values from approximate graph readings "
    "and calculations derived from them. Do not invent values or present "
    "approximate estimates as verified manual values. If the context "
    "does not establish a requested value and no graph-reading guidance "
    "is supplied, say so. Show calculations only when their inputs are "
    "supported. Identify approximate results as approximate. "
    "Cite the printed page and figure when available. "
    "Do not invent additional specifications."
)

QUESTIONS = [
    (
        "Q1 - Formula",
        "What formula does Figure 38 use to convert measured insulation "
        "resistance to its equivalent value at 40°C? Explain the variables."
    ),
    (
        "Q2 - Printed example",
        "What exact numerical example does the ABB manual print for "
        "converting insulation resistance to its equivalent value at 40°C?"
    ),
    (
	"Q3 - Approximate calculation",
    	"If insulation resistance is measured as 50 MΩ at 30°C, can we "
    	"estimate its equivalent value at 40°C using Figure 38? For this "
    	"test, use k approximately equal to 0.5 at 30°C as an approximate "
    	"graph reading, not an exact printed value. Show the calculation "
    	"using R40 = k x R, and explain whether the result is an exact "
    	"printed value or an approximate graph-based estimate."
    ),
]


def main():
    print("Collection:", COLLECTION_NAME)
    print("Points available:", info.points_count)

    for label, query in QUESTIONS:
        print("\n" + "=" * 78)
        print(label)
        print("QUESTION:", query)

        # Retrieve both the original text and the figure description.
        results = vectorstore.similarity_search_with_score(query, k=2)

        if not results:
            print("No context retrieved. Skipping answer generation.")
            continue

        contexts = []
        for document, score in results:
            contexts.append(document.page_content)
            print(f"\nRetrieved score: {score:.4f}")
            print("Retrieved metadata:", document.metadata)

        retrieved_context = "\n\n--- Retrieved document ---\n\n".join(contexts)

        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                f"Question: {query}\n\n"
                f"Retrieved ABB manual context:\n{retrieved_context}",
            ),
        ])

        print("\nGENERATED ANSWER:")
        print(response.content)

    print("\n" + "=" * 78)
    print("Answer-generation test completed.")
    print("This script only retrieves from the test collection.")
    print("It does not write to Qdrant or modify the document registry.")


if __name__ == "__main__":
    main()
