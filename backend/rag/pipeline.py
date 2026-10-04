"""6-stage RAG pipeline: query rewrite -> retrieve -> BM25 rerank -> assemble -> LLM -> citations."""
import asyncio
import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

from rag.llm import chat_llm
from rag.qdrant_init import get_qdrant_client
from rag.vectorstore import QDRANT_COLLECTION_NAME, create_or_load_vectorstore

load_dotenv()

_RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "15"))
_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

_SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. "
    "Answer ONLY using the provided context — this is your sole grounding source. "
    "Do not hallucinate or add information not present in the context. "
    "If the context does not fully answer the question, say clearly what is missing "
    "rather than filling the gap with outside knowledge."
)

# Answer personas — the user picks one when starting a chat. Each sets the level
# of detail of the answer; the grounding and citation rules apply to all of them.
PERSONAS: dict[str, str] = {
    "technician": (
        "The reader is a field technician at the machine who needs to act now. Be brief: "
        "lead with what to do as a short numbered list of steps, state key values and safety "
        "warnings, and leave out background theory. Aim for about 120 words or fewer."
    ),
    "engineer": (
        "The reader is a maintenance engineer. Give a standard level of detail: a direct "
        "answer first, then the likely causes or reasoning and the procedure, with the "
        "relevant values, limits, and conditions."
    ),
    "specialist": (
        "The reader is a reliability specialist doing in-depth analysis. Be detailed and "
        "complete: cover every relevant cause, condition, limit, exception, and related "
        "check found in the context, organised under headings, with tables for values or "
        "comparisons."
    ),
}
DEFAULT_PERSONA = "engineer"


def persona_instructions(persona: str | None) -> str:
    return PERSONAS.get(persona or "", PERSONAS[DEFAULT_PERSONA])


_RAG_TEMPLATE = """\
Context:
{context}

Question:
{question}

Instructions:
- Answer based solely on the context above — do not use outside knowledge.
- Audience and level of detail: {persona_instructions}
- Each context passage starts with a numbered label giving its source, e.g.
  "[2] Document Name, p. 12 · Section > Subsection". Citations are mandatory: end
  every factual sentence, list item, or table row with the marker(s) of the
  passage(s) it came from, e.g. "[2]" or "[1][3]". Use only numbers that appear in
  the context labels — never guess or invent one — and write nothing else of the
  label (no document name, page, or section) inside the brackets. Do not add a
  marker to a sentence that only says the context is insufficient, and do not add
  a separate sources/references list.
- Format the answer in Markdown. These structure rules apply at every level of detail:
  - When facts you use come from a table in the context, present them as a Markdown
    table with the relevant rows and columns — never rewrite table data as prose.
  - Present a procedure or sequence as a numbered list, one action per item, and
    any other enumeration (causes, checks, conditions) as a bullet list.
  - Use headings only when the answer has more than one part. Do not force
    structure onto a one-sentence answer.
- Diagrams: a passage may be followed by "Figures for this passage", each with a
  caption and an image path. When one of them illustrates an instruction you give,
  show it on its own line directly after that instruction, as a Markdown image
  using exactly the caption and path from the context:
  ![Figure 7 Lifting the machine](diagrams/abc/fig7_1.png)
  Show a figure at most once, only when it is relevant to the question, and never
  write an image path that is not in the context.
- Write any mathematical formula in LaTeX using $...$ for inline math and $$...$$
  for a standalone/display formula (not \( \) or \[ \]).
- If the context does not contain enough information, say so clearly.
"""


# ── Stage 1: Query Processing ─────────────────────────────────────────────────

def _stage1_process_query(query: str, llm: ChatOpenAI, metadata: dict | None = None) -> str:
    """Clean, rewrite, and disambiguate the user query for semantic search."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a query optimizer for a knowledge base. "
                "Rewrite the user query to be clear, specific, and optimised for semantic search. "
                "Remove filler words, fix typos, and make implicit intent explicit. "
                "Return ONLY the rewritten query — no explanation.",
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {"query": query},
        config={"run_name": "query_rewrite", "tags": ["rag", "stage1"], "metadata": metadata or {}},
    ).strip()


# ── Stage 2: Retrieval ────────────────────────────────────────────────────────

def _stage2_retrieve(query: str) -> list[Document]:
    """Semantic search against Qdrant — fetches a larger candidate pool for BM25 reranking."""
    vectorstore = create_or_load_vectorstore()
    return vectorstore.similarity_search(query, k=_RETRIEVAL_TOP_K)


# ── Stage 3: Hybrid Reranking (semantic rank + BM25 rank) ─────────────────────

# Words, numbers and codes ("6220", "pt-100", "7.4.3") without the punctuation or
# Markdown table pipes around them, so "| 6220 |" and "machine?" match the query.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# Reciprocal Rank Fusion constant — the standard value; damps the gap between adjacent ranks.
_RRF_K = 60
# Semantic rank counts more than keyword rank: BM25 over ~15 candidates is the noisier signal.
_SEMANTIC_WEIGHT = float(os.getenv("RERANK_SEMANTIC_WEIGHT", "2"))


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stage3_rerank(query: str, chunks: list[Document]) -> list[Document]:
    """Rerank the semantic-search candidates by fusing their semantic rank with a BM25 rank.

    BM25 (Okapi) scores exact keyword overlap, which helps with specific codes,
    names and technical terms — but on its own it favours long prose that
    repeats the query's words over a compact table that actually holds the
    answer. Reciprocal Rank Fusion combines the two orderings, so a chunk has
    to do well on meaning or keywords, and best on both, to reach the top.

    `chunks` must be in semantic-similarity order (best first), as returned by stage 2.
    """
    if not chunks:
        return []

    seen: set[str] = set()
    unique: list[Document] = []
    for doc in chunks:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)

    bm25 = BM25Okapi([_tokenize(doc.page_content) for doc in unique])
    scores = bm25.get_scores(_tokenize(query))

    bm25_order = sorted(range(len(unique)), key=lambda i: scores[i], reverse=True)
    bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order, start=1)}
    fused = {
        idx: _SEMANTIC_WEIGHT / (_RRF_K + semantic_rank) + 1 / (_RRF_K + bm25_rank[idx])
        for semantic_rank, idx in enumerate(range(len(unique)), start=1)
    }
    # sorted() is stable, so ties keep semantic order.
    top_indices = sorted(fused, key=fused.get, reverse=True)[:_RERANK_TOP_N]

    reranked: list[Document] = []
    for idx in top_indices:
        doc = unique[idx]
        doc.metadata["bm25_score"] = round(float(scores[idx]), 4)
        reranked.append(doc)

    return reranked


# ── Stage 4: Context Assembly ─────────────────────────────────────────────────

def _clean_title(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ").replace("-", " ").strip()


def _page_number(doc: Document) -> int | None:
    """Best-effort 1-indexed page number, from whichever ingestion pipeline produced the chunk.

    Docling chunks and MarkItDown PDF chunks (the standard pipeline) carry
    `page_no` (already 1-indexed). Chunks ingested earlier by PyPDFLoader carry
    `page` (0-indexed), so it's shifted by one. Chunks from non-paginated
    sources (.docx, .pptx, .txt, ...) have neither — page is unknown.
    """
    page_no = doc.metadata.get("page_no")
    if isinstance(page_no, int):
        return page_no
    page = doc.metadata.get("page")
    if isinstance(page, int):
        return page + 1
    return None


def _chunk_type(doc: Document) -> str | None:
    """Chunk type worth surfacing in a citation — plain text/prose is the default and omitted."""
    chunk_type = doc.metadata.get("chunk_type")
    return chunk_type if chunk_type in ("table", "figure", "note", "warning") else None


_MAX_FIGURES_PER_CHUNK = 3


def _figures_for_chunks(chunks: list[Document]) -> list[list[dict]]:
    """Figures to offer alongside each chunk, parallel to `chunks`.

    A figure chunk offers itself; a text/table chunk offers the figures its
    text mentions (`figure_refs`, set at ingestion), looked up among the same
    document's figure chunks in Qdrant. Each figure is
    {"number", "caption", "image_paths"}.
    """
    wanted_docs = {
        doc.metadata.get("document_id")
        for doc in chunks
        if doc.metadata.get("figure_refs") and doc.metadata.get("document_id")
    }
    stored: dict[tuple[str, int], dict] = {}
    if wanted_docs:
        try:
            points, _ = get_qdrant_client().scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="metadata.chunk_type", match=MatchValue(value="figure")),
                        FieldCondition(key="metadata.document_id", match=MatchAny(any=sorted(wanted_docs))),
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"⚠️ Figure lookup failed, answering without diagrams: {e}")
            points = []
        for point in points:
            meta = (point.payload or {}).get("metadata", {})
            if meta.get("image_paths") and meta.get("figure_number") is not None:
                stored[(meta.get("document_id"), meta["figure_number"])] = meta

    def as_figure(meta: dict) -> dict:
        return {
            "number": meta.get("figure_number"),
            "caption": meta.get("figure_caption", ""),
            "image_paths": meta.get("image_paths", []),
        }

    result: list[list[dict]] = []
    for doc in chunks:
        meta = doc.metadata
        if meta.get("chunk_type") == "figure" and meta.get("image_paths"):
            result.append([as_figure(meta)])
            continue
        figures = [
            as_figure(stored[(meta.get("document_id"), number)])
            for number in meta.get("figure_refs") or []
            if (meta.get("document_id"), number) in stored
        ]
        result.append(figures[:_MAX_FIGURES_PER_CHUNK])
    return result


def _stage4_assemble_context(chunks: list[Document], figures: list[list[dict]] | None = None) -> str:
    """Numbers each chunk [n] — the same n the answer cites and stage 6 reports —
    and lists the figures (caption + image path) the answer may show with it."""
    parts = []
    for number, doc in enumerate(chunks, start=1):
        label = _clean_title(doc.metadata.get("document_name", "Unknown"))
        page = _page_number(doc)
        if page is not None:
            label += f", p. {page}"
        section_path = doc.metadata.get("section_path")
        if section_path:
            label += f" · {section_path}"
        chunk_type = _chunk_type(doc)
        if chunk_type:
            label += f" ({chunk_type})"
        part = f"[{number}] {label}\n{doc.page_content}"
        chunk_figures = figures[number - 1] if figures else []
        if chunk_figures:
            lines = [
                f"- {figure['caption']} — image path: {path}"
                for figure in chunk_figures
                for path in figure["image_paths"]
            ]
            part += "\n\nFigures for this passage:\n" + "\n".join(lines)
        parts.append(part)
    return "\n\n---\n\n".join(parts)


# ── Stage 6: Response Generation ──────────────────────────────────────────────

def _stage6_build_citations(chunks: list[Document]) -> list[dict]:
    """One structured citation per BM25-reranked chunk, numbered to match the
    [n] labels of stage 4. Not deduplicated — select_cited() narrows the list
    to the numbers the answer actually used."""
    citations: list[dict] = []
    for i, doc in enumerate(chunks):
        citations.append(
            {
                "number": i + 1,
                "document_id": doc.metadata.get("document_id", ""),
                "document_name": _clean_title(doc.metadata.get("document_name", "Unknown")),
                "chunk_index": i,
                "bm25_score": doc.metadata.get("bm25_score"),
                "page": _page_number(doc),
                "section_path": doc.metadata.get("section_path") or None,
                "chunk_type": _chunk_type(doc),
                "image_path": doc.metadata.get("image_path") or next(iter(doc.metadata.get("image_paths") or []), None),
            }
        )
    return citations


_CITATION_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def select_cited(answer: str, citations: list[dict]) -> list[dict]:
    """Citations whose [n] marker appears in the finished answer, in number order."""
    cited = {int(n) for group in _CITATION_MARKER_RE.findall(answer) for n in group.split(",")}
    return [c for c in citations if c["number"] in cited]


# ── Public entry points (stages 1-4+6 upfront, stage 5 token-by-token or invoked) ──

def _prepare_context_sync(query: str, metadata: dict | None = None) -> tuple[str, list[dict], str]:
    """Stages 1-4 + citations (stage 6). Returns (context, citations, rewritten_query)."""
    llm = chat_llm()
    rewritten = _stage1_process_query(query, llm, metadata)

    candidates = _stage2_retrieve(rewritten)
    if not candidates:
        return "", [], rewritten

    reranked_chunks = _stage3_rerank(rewritten, candidates)
    context = _stage4_assemble_context(reranked_chunks, _figures_for_chunks(reranked_chunks))
    citations = _stage6_build_citations(reranked_chunks)
    return context, citations, rewritten


async def prepare_context(query: str, metadata: dict | None = None) -> tuple[str, list[dict], str]:
    """Async wrapper for stages 1-4+6, run in a thread pool.

    `metadata` (e.g. conversation_id, user_id) is attached to LangSmith traces
    when LANGCHAIN_TRACING_V2 is enabled.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _prepare_context_sync, query, metadata)


def build_answer_chain():
    """LCEL chain for stage 5 that supports token-by-token .astream()."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _RAG_TEMPLATE),
        ]
    )
    llm = chat_llm(streaming=True)
    return prompt | llm | StrOutputParser()
