
from pip._vendor import truststore as t
t.extract_from_ssl()

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from rag.qdrant_init import get_qdrant_client

load_dotenv(Path(__file__).resolve().parent / ".env")

COLLECTION_NAME = "imka_vlm_test"
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIMENSION,
)

vectorstore = QdrantVectorStore(
    client=get_qdrant_client(),
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

query = (
"What is the formula used in Figure 38 to convert "
"measured insulation resistance to its equivalent "
"value at 40°C? Explain each variable."
)

results = vectorstore.similarity_search_with_score(query, k=1)

if not results:
    raise RuntimeError("No figure description was retrieved.")

document, score = results[0]

print(f"Retrieved similarity score: {score}")
print("\nRetrieved context:\n")
print(document.page_content)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

response = llm.invoke([
    (
        "system",
        "Answer using only the retrieved ABB manual context. "
        "Clearly distinguish exact printed values, approximate graph readings, "
        "and calculations derived from them. Do not invent values or present "
        "interpolated estimates as verified manual values. If the context does "
        "not establish a requested value, state that it cannot be confirmed "
        "from the retrieved context. Show calculations only when their inputs "
        "are supported, and identify approximate results as approximate. "
        "Cite the printed page and figure when available. "
        "Do not invent additional specifications."
    ),
    (
        "human",
        f"Question: {query}\n\nRetrieved context:\n{document.page_content}"
    ),
])

print("\n" + "=" * 70)
print("GENERATED RAG ANSWER")
print("=" * 70)
print(response.content)
