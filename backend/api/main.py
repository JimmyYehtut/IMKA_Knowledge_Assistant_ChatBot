import hashlib
import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from ingest.embed_and_store import ingest_files
from ingest.embed_and_store_docling import ingest_files_docling
from ingest.load_documents import SUPPORTED_EXTENSIONS
from ingest.registry import get_document, list_documents, remove_document
from rag.qdrant_init import get_qdrant_client
from rag.topics import get_topics
from rag.vectorstore import QDRANT_COLLECTION_NAME

from . import auth as auth_module
from . import chat as chat_module
from . import feedback as feedback_module
from . import openai_compat
from .database import create_tables
from .schemas import HealthResponse, IngestResponse

load_dotenv()

ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS
DOCLING_ALLOWED_EXTENSIONS = {".pdf"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield


app = FastAPI(title="RAG Data Pipeline API", lifespan=lifespan)

allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_module.router)
app.include_router(chat_module.router)
app.include_router(feedback_module.router)
app.include_router(openai_compat.router)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
(DATA_DIR / "diagrams").mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/api/health", response_model=HealthResponse)
def health():
    try:
        get_qdrant_client().get_collections()
        qdrant_reachable = True
    except Exception:
        qdrant_reachable = False

    return HealthResponse(
        status="ok",
        openai_configured=bool(os.getenv("OPENAI_API_KEY")),
        qdrant_reachable=qdrant_reachable,
    )


@app.get("/api/documents", response_model=list[dict])
def get_documents():
    return list_documents()


@app.get("/api/topics", response_model=list[dict])
def get_document_topics(persona: str | None = None):
    """Topics the ingested documents cover, optionally only those suited to one answer persona."""
    return get_topics(persona)


@app.post("/api/documents", response_model=IngestResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    version: str = Form("1.0"),
    metadata: str = Form("{}"),
    pipeline: str = Form("standard"),
):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY must be configured on the server.")

    if pipeline not in ("standard", "docling"):
        raise HTTPException(status_code=400, detail="pipeline must be 'standard' or 'docling'.")

    try:
        metadata_dict = json.loads(metadata)
        if not isinstance(metadata_dict, dict) or not all(isinstance(v, str) for v in metadata_dict.values()):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object of string values.")

    allowed_extensions = DOCLING_ALLOWED_EXTENSIONS if pipeline == "docling" else ALLOWED_EXTENSIONS

    temp_paths = []
    original_names = []
    temp_dir = tempfile.gettempdir()

    try:
        for upload in files:
            ext = os.path.splitext(upload.filename or "")[1].lower()
            if ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or '(none)'} for {upload.filename}")

            file_hash = hashlib.md5(upload.filename.encode("utf-8")).hexdigest()
            temp_path = os.path.join(temp_dir, f"uploaded_{file_hash}{ext}")
            content = await upload.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            temp_paths.append(temp_path)
            original_names.append(upload.filename)

        try:
            ingest_fn = ingest_files_docling if pipeline == "docling" else ingest_files
            # Ingestion is blocking and can take minutes (Docling parsing, embedding);
            # run it off the event loop so chat and other requests keep being served.
            result = await run_in_threadpool(
                ingest_fn,
                temp_paths,
                metadata=metadata_dict,
                version=version,
                original_names=original_names,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e

        if not result:
            raise HTTPException(status_code=422, detail="No text could be extracted from the uploaded file(s).")

        documents = [get_document(rec["document_id"]) for rec in result["documents"]]
        return IngestResponse(total_chunks=result["total_chunks"], documents=documents)
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    record = get_document(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {e}") from e

    shutil.rmtree(DATA_DIR / "diagrams" / document_id, ignore_errors=True)
    remove_document(document_id)
    return {"deleted": True, "document_id": document_id}
