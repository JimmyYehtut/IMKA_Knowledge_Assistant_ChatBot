import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from .qdrant_init import ensure_collection, get_qdrant_client

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower() or "openai"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "ragindex")


def embedding_model_name() -> str:
    """The embedding model behind get_embeddings(), as reported in evaluation traces."""
    if EMBEDDING_PROVIDER == "qwen":
        return os.getenv("QWEN_EMBED_MODEL_ALIAS", "Qwen3-Embedding-4B-Q4_K_M")
    return EMBEDDING_MODEL


def get_embeddings():
    """
    Return the embeddings selected by EMBEDDING_PROVIDER: OpenAI (default), or
    Qwen3-Embedding-4B on the RunPod endpoint ("qwen"). Both are cut to
    EMBEDDING_DIMENSION where the model allows it, which must match the collection.
    """
    if EMBEDDING_PROVIDER == "qwen":
        from .qwen_client import QwenRunpodEmbeddings
        return QwenRunpodEmbeddings(dimensions=EMBEDDING_DIMENSION)
    if EMBEDDING_MODEL in ["text-embedding-3-small", "text-embedding-3-large"]:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION)
    else:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def create_or_load_vectorstore(documents=None):
    embeddings = get_embeddings()
    client = get_qdrant_client()
    ensure_collection(client, QDRANT_COLLECTION_NAME, EMBEDDING_DIMENSION)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )

    if documents:
        vectorstore.add_documents(documents)

    return vectorstore
