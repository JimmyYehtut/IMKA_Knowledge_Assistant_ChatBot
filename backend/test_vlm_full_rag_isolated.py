

from pip._vendor import truststore as t
t.extract_from_ssl()

"""Read-only full RAG pipeline test against the isolated VLM test collection.

Does not call the API route, persist chat history, ingest documents,
or write to Qdrant.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

import rag.pipeline as pipeline
from rag.qdrant_init import get_qdrant_client
from rag.vectorstore import EMBEDDING_DIMENSION, EMBEDDING_MODEL

TEST_COLLECTION = "imka_vlm_docling_test"
PRODUCTION_COLLECTION = "ragindex"


def safe_test_retrieve(query: str):
    """Search only the isolated test collection."""
    client = get_qdrant_client()

    collections = {
        item.name for item in client.get_collections().collections
    }
    if TEST_COLLECTION not in collections:
        raise RuntimeError(f"Missing test collection: {TEST_COLLECTION}")

    if TEST_COLLECTION == PRODUCTION_COLLECTION:
        raise RuntimeError("Safety check failed: test and production collections match")

    count = client.count(
        collection_name=TEST_COLLECTION,
        exact=True,
    ).count
    if count == 0:
        raise RuntimeError("Test collection is empty; refusing to continue")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=TEST_COLLECTION,
        embedding=embeddings,
    )
    return vectorstore.similarity_search(
        query,
        k=min(pipeline._RETRIEVAL_TOP_K, count),
    )


# Temporary in-memory patches; no project files are changed.
pipeline._stage2_retrieve = safe_test_retrieve

# Keep figure lookup inside the isolated collection.
pipeline.QDRANT_COLLECTION_NAME = TEST_COLLECTION


def run_question(question: str) -> None:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    context, citations, rewritten = pipeline._prepare_context_sync(question)

    print("\nREWRITTEN QUERY:")
    print(rewritten)

    print("\nASSEMBLED CONTEXT:")
    print(context if context else "[No relevant context retrieved]")

    print("\nGENERATING ANSWER:")
    if not context:
        print("No relevant documents found in the isolated test collection.")
        return

    chain = pipeline.build_answer_chain()
    answer = chain.invoke(
        {
            "context": context,
            "question": rewritten,
            "persona_instructions": pipeline.persona_instructions("engineer"),
        }
    )

    print("\nANSWER:")
    print(answer)

    print("\nSELECTED CITATIONS:")
    for citation in pipeline.select_cited(answer, citations):
        print(citation)


if __name__ == "__main__":
    client = get_qdrant_client()

    before_test = {
        name: client.count(collection_name=name, exact=True).count
        for name in (TEST_COLLECTION, PRODUCTION_COLLECTION)
    }
    print("Point counts before test:", before_test)

    questions = [
        "What is the formula for converting insulation resistance to the equivalent value at 40 degrees Celsius?",
        "What example does ABB give for insulation resistance measured at 20 degrees Celsius?",
        "If the measured insulation resistance is 50 megaohms at 30 degrees Celsius, what is the approximate equivalent resistance at 40 degrees Celsius?",
    ]

    for question in questions:
        run_question(question)

    after_test = {
        name: client.count(collection_name=name, exact=True).count
        for name in (TEST_COLLECTION, PRODUCTION_COLLECTION)
    }
    print("\nPoint counts after test:", after_test)

    if before_test != after_test:
        raise RuntimeError(
            "Collection point counts changed during the test. "
            f"Before={before_test}; after={after_test}"
        )

    print("\nPASS: Qdrant point counts are unchanged.")
