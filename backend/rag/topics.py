"""Topic catalogue: what the ingested documents cover, and which answer persona each topic suits.

Topics are the documents' own second-level section headings ("7.5 Maintenance
of bearings ...") read from chunk metadata in Qdrant, so the list always
reflects what was ingested. The chat model assigns each topic to personas
once; the result is cached on disk until the set of ingested documents changes.
"""
import json
import re
import threading
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ingest.registry import list_documents
from rag.llm import chat_llm
from rag.pipeline import PERSONAS
from rag.qdrant_init import get_qdrant_client
from rag.vectorstore import QDRANT_COLLECTION_NAME

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "topics_cache.json"

_SECTION_RE = re.compile(r"^(\d+)\.(\d+)\s+(.+)$")  # "7.5 Title" — second-level headings only
_CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s+(.+)$", re.IGNORECASE)
_GENERIC_TITLES = {"general", "general information", "introduction", "important note"}

_lock = threading.Lock()

_AUDIENCE_PROMPT = (
    "You are organising the table of contents of technical maintenance documents for a "
    "knowledge assistant with three reader personas:\n"
    "- technician: a field technician working at the machine — handling, inspection, operation, "
    "routine maintenance, troubleshooting, safety precautions.\n"
    "- engineer: a maintenance engineer — installation, alignment, connections, commissioning, "
    "maintenance planning, troubleshooting.\n"
    "- specialist: a reliability specialist — every technical topic, including design limits, "
    "standards, life cycle, and analysis.\n\n"
    "For each numbered topic, list the personas whose work it is relevant to. A topic can suit "
    "several personas. Return one entry per topic, using the topic's number as `index`."
)


class _TopicAudience(BaseModel):
    index: int
    personas: list[Literal["technician", "engineer", "specialist"]]


class _Audiences(BaseModel):
    topics: list[_TopicAudience] = Field(description="One entry per topic in the input list.")


def _scan_sections() -> list[dict]:
    """(document_id, document_name, section_path, page_no) for every chunk in the collection."""
    client = get_qdrant_client()
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        return []
    fields = ["document_id", "document_name", "section_path", "page_no"]
    rows, offset = [], None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=[f"metadata.{f}" for f in fields],
            with_vectors=False,
        )
        rows.extend((p.payload or {}).get("metadata", {}) for p in points)
        if offset is None:
            return rows


def _extract_topics(rows: list[dict]) -> list[dict]:
    """Second-level numbered headings per document, in document order, with their chapter and first page."""
    chapters: dict[tuple[str, int], str] = {}
    topics: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        section = (row.get("section_path") or "").strip()
        doc_id = row.get("document_id", "")
        page = row.get("page_no")

        chapter = _CHAPTER_RE.match(section)
        if chapter:
            chapters[(doc_id, int(chapter.group(1)))] = chapter.group(2).strip()
            continue

        match = _SECTION_RE.match(section)
        if not match or match.group(3).strip().lower() in _GENERIC_TITLES:
            continue
        key = (doc_id, int(match.group(1)), int(match.group(2)))
        topic = topics.setdefault(
            key,
            {
                "title": match.group(3).strip(),
                "chapter_number": key[1],
                "document_id": doc_id,
                "document_name": row.get("document_name", ""),
                "page": page,
            },
        )
        if isinstance(page, int) and (topic["page"] is None or page < topic["page"]):
            topic["page"] = page

    ordered = [topics[key] for key in sorted(topics)]
    for topic in ordered:
        number = topic.pop("chapter_number")
        topic["chapter"] = chapters.get((topic["document_id"], number), f"Section {number}")
    return ordered


def _assign_personas(topics: list[dict]) -> None:
    """Adds `personas` to each topic. Falls back to all personas if the model call fails."""
    for topic in topics:
        topic["personas"] = list(PERSONAS)
    if not topics:
        return
    listing = "\n".join(f"{i}. {t['chapter']} — {t['title']}" for i, t in enumerate(topics))
    prompt = ChatPromptTemplate.from_messages([("system", _AUDIENCE_PROMPT), ("human", "{topics}")])
    try:
        result = (prompt | chat_llm().with_structured_output(_Audiences)).invoke(
            {"topics": listing}, config={"run_name": "topic_persona_assignment", "tags": ["topics"]}
        )
    except Exception as e:
        print(f"⚠️ Topic persona assignment failed, showing all topics to every persona: {e}")
        return
    for entry in result.topics:
        if 0 <= entry.index < len(topics) and entry.personas:
            topics[entry.index]["personas"] = entry.personas


def _documents_signature() -> str:
    return ",".join(sorted(doc["document_id"] for doc in list_documents()))


def get_topics(persona: str | None = None) -> list[dict]:
    """Topic catalogue of the ingested documents, optionally narrowed to one persona."""
    signature = _documents_signature()
    with _lock:
        cached = None
        if CACHE_PATH.exists():
            try:
                cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except ValueError:
                cached = None
        if cached and cached.get("signature") == signature:
            topics = cached["topics"]
        else:
            topics = _extract_topics(_scan_sections())
            _assign_personas(topics)
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(json.dumps({"signature": signature, "topics": topics}, indent=2), encoding="utf-8")

    if persona in PERSONAS:
        topics = [t for t in topics if persona in t["personas"]]
    return topics
