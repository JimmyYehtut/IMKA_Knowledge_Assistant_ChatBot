"""Evaluation helpers for runs test-A … test-H (report/notes.md, "Evaluation Plan").

A run answers the whole question set against one configuration's five documents (one
document_id per PDF, all searchable at once), three times over, then scores:

  retrieval  Hit@K, MRR, correct-PDF rate in the top 5 — from the `rag` trace the backend
             returns. A retrieved chunk is relevant when it comes from the question's PDF and
             sits on (or next to) a page named in its `source`. No LLM, deterministic.
  judge      RAGAS through Claude: Haiku 5.5 for faithfulness, answer_relevancy and
             context_recall, Sonnet 5.5 for answer_accuracy (Haiku scored accuracy 0.22 lower
             than Sonnet in the dry run). `check=True` also scores context_recall with Sonnet,
             for the Haiku-vs-Sonnet agreement figure. Out-of-scope questions get one Haiku
             refusal check instead.
  compare    paired bootstrap 95% CI of run B − run A on the questions both answered, per
             question averaged over the repeats.

Every step checkpoints to ../output/<run>/ after each question and skips work already done
(matched on question text), so a re-run resumes instead of paying again.

Runs on test/.venv (see requirements.txt), not on backend/.venv.
"""
import asyncio
import json
import math
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# pip-system-certs (if installed) injects pip's vendored truststore into `ssl`, and the
# OpenAI client's own truststore then recurses forever on the TLS handshake ("Connection
# error."). Undo that injection before any client is created.
try:
    from pip._vendor import truststore as _pip_truststore

    _pip_truststore.extract_from_ssl()
except Exception:
    pass

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

HERE = Path(__file__).resolve().parent          # test/code
ROOT = HERE.parent.parent                       # repository root
BACKEND = ROOT / "backend"
OUTPUT_DIR = HERE.parent / "output"             # test/output
QUESTIONS_PATH = HERE / "questions.json"

# test/code/.env: IMKA_API_URL, IMKA_EMAIL, IMKA_PASSWORD. backend/.env: ANTHROPIC_API_KEY,
# OPENAI_API_KEY (answer_relevancy embeddings). load_dotenv never overrides, so test/code/.env wins.
load_dotenv(HERE / ".env")
load_dotenv(BACKEND / ".env")

DEFAULT_API_URL = "http://localhost:8000"
ANSWER_TIMEOUT_S = 900.0
REPEATS = (1, 2, 3)

# ── The plan ──────────────────────────────────────────────────────────────────

# Registry flags of each ingestion configuration.
CONFIGS = {
    "C1": {"pipeline": "standard", "chunking": False, "captioning": False},
    "C2": {"pipeline": "docling", "chunking": True, "captioning": False},
    "C3": {"pipeline": "docling", "chunking": True, "captioning": True},
}
MINI = "gpt-4o-mini"
LUNA = "gpt-5.6-luna"
QWEN = "Qwen3.8-27B-UD-Q4_K_XL"
DEFAULT_EMBEDDING = "text-embedding-3-small"
QWEN_EMBEDDING = "Qwen3-Embedding-4B-Q4_K_M"

# Each stack uses one model for everything: the C3 figure descriptions come from the same
# model that answers (caption_model). test-H is the hybrid strategy: the strong model (luna)
# does all ingestion, the cheap model (gpt-4o-mini) answers. It also splits test-C → test-F
# into what the ingestion model and what the answering model contribute.
# answer_model: what the backend must report (OPENAI_CHAT_MODEL, or the Qwen alias with
# LLM_PROVIDER=qwen). "document_ids" may be added to pin a run's documents by hand.
RUNS = {
    "test-A": {"config": "C1", "answer_model": MINI},
    "test-B": {"config": "C2", "answer_model": MINI},
    "test-C": {"config": "C3", "answer_model": MINI, "caption_model": MINI},
    "test-D": {"config": "C1", "answer_model": LUNA},
    "test-E": {"config": "C2", "answer_model": LUNA},
    "test-F": {"config": "C3", "answer_model": LUNA, "caption_model": LUNA},
    "test-G": {"config": "C3", "answer_model": QWEN, "caption_model": QWEN, "embedding": QWEN_EMBEDDING},
    "test-H": {"config": "C3", "answer_model": MINI, "caption_model": LUNA},
}

GROUPS = ("tables", "topical", "figures", "cross_document", "paraphrase", "out_of_scope")
IN_SCOPE = tuple(g for g in GROUPS if g != "out_of_scope")

# (run A, run B, primary metric, question groups) — fixed before the runs (23 Oct 2026).
COMPARISONS = [
    # Techniques, within each API stack
    ("test-A", "test-B", "context_recall", ("tables", "topical")),   # chunking, gpt-4o-mini
    ("test-D", "test-E", "context_recall", ("tables", "topical")),   # chunking, luna
    ("test-B", "test-C", "answer_accuracy", ("figures",)),            # captioning, gpt-4o-mini's own
    ("test-E", "test-F", "answer_accuracy", ("figures",)),            # captioning, luna's own
    ("test-B", "test-H", "answer_accuracy", ("figures",)),            # captioning by luna, gpt-4o-mini answering
    # Models, at each configuration
    ("test-A", "test-D", "answer_accuracy", IN_SCOPE),                # answering model only (C1)
    ("test-B", "test-E", "answer_accuracy", IN_SCOPE),                # answering model only (C2)
    ("test-C", "test-F", "answer_accuracy", IN_SCOPE),                # whole stack (C3)
    ("test-C", "test-H", "answer_accuracy", ("figures",)),            # ingestion model only (hybrid)
    ("test-H", "test-F", "answer_accuracy", IN_SCOPE),                # answering model only (hybrid -> luna)
    ("test-A", "test-D", "refused", ("out_of_scope",)),
    # API stack vs fully local stack: captioner, embedding and answering model all change.
    ("test-F", "test-G", "answer_accuracy", IN_SCOPE),
]

SOURCE_PAGE = re.compile(r"\bp\.\s*(\d+)")
TOLERANCES = {"exact": 0, "within_1_page": 1}
DEFAULT_TOLERANCE = "within_1_page"
DEFAULT_K = (1, 3, 5)


def expects_refusal(question: dict) -> bool:
    """Only out-of-scope questions: none of the PDFs answers them. In-scope "absence" and
    "deferred-value" probes have a real reference ("no fixed interval; see the bearing
    plate"), so they are scored on answer accuracy like every other in-scope question."""
    return question.get("group") == "out_of_scope"


def run_dir(run: str) -> Path:
    if run not in RUNS:
        raise ValueError(f"Unknown run {run!r}. Runs: {', '.join(RUNS)}")
    return OUTPUT_DIR / run


def _write_json(path: Path, payload: dict) -> None:
    """Write via a temp file and rename, so a crash never leaves half a JSON file.

    On Windows the write or the rename can fail while another process (an editor, a
    reader, a virus scanner) has a file open — PermissionError, or OSError "Invalid
    argument" — so both are retried for a few seconds before giving up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for attempt in range(20):
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            return
        except OSError:  # PermissionError is a subclass
            if attempt == 19:
                raise
            time.sleep(0.25)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  !! {path.name} is not valid JSON — ignoring it")
        return {}


# ── Backend client ────────────────────────────────────────────────────────────

class Backend:
    """Login, document registry and chat completions on the running IMKA backend."""

    def __init__(self, api_url: str, email: str, password: str, timeout: float = ANSWER_TIMEOUT_S):
        self.api_url = api_url.rstrip("/")
        self.email, self.password = email, password
        self.http = httpx.Client(base_url=self.api_url, timeout=httpx.Timeout(timeout, connect=10.0))
        self.token: str | None = None
        self._login_lock = threading.Lock()

    def connect(self) -> "Backend":
        try:
            self.http.get("/v1/models").raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Backend not reachable at {self.api_url} ({type(e).__name__}: {e}). "
                               "Start it: cd backend && uvicorn api.main:app --port 8000") from e
        self.login()
        return self

    def login(self) -> None:
        if not self.email or not self.password:
            raise RuntimeError("Set IMKA_EMAIL and IMKA_PASSWORD in test/code/.env (an account that exists in the IMKA UI).")
        with self._login_lock:
            r = self.http.post("/api/auth/login", json={"email": self.email, "password": self.password})
            if r.status_code == 401:
                raise RuntimeError(f"Login failed for {self.email}: invalid credentials.")
            r.raise_for_status()
            self.token = r.json()["access_token"]

    def documents(self) -> list[dict]:
        r = self.http.get("/api/documents")
        r.raise_for_status()
        return r.json()

    def ask(self, question: str, document_ids: list[str]) -> dict:
        body = {
            "messages": [{"role": "user", "content": question}],
            "include_context": True,
            "document_ids": document_ids,
        }
        for attempt in (1, 2):
            r = self.http.post("/v1/chat/completions", json=body,
                               headers={"Authorization": f"Bearer {self.token}"})
            if r.status_code == 401 and attempt == 1:  # token expired mid-run — log in again once
                self.login()
                continue
            if r.is_error:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise RuntimeError("unauthorized after re-login")


def connect(api_url: str | None = None) -> Backend:
    import os

    return Backend(api_url or os.getenv("IMKA_API_URL") or DEFAULT_API_URL,
                   os.getenv("IMKA_EMAIL", ""), os.getenv("IMKA_PASSWORD", "")).connect()


# ── Questions ─────────────────────────────────────────────────────────────────

def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """All questions; each needs `question`, `group`, `answer`, and (in scope) `document_name` and `source`."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    seen = set()
    for i, q in enumerate(questions, 1):
        if q.get("group") not in GROUPS:
            problems.append(f"#{i}: group {q.get('group')!r} is not one of {GROUPS}")
        if q.get("group") != "out_of_scope" and not q.get("document_name"):
            problems.append(f"#{i}: in-scope question without document_name")
        if q["question"] in seen:
            problems.append(f"#{i}: duplicate question text")
        seen.add(q["question"])
    if problems:
        raise RuntimeError(f"{Path(path).name} has problems:\n  " + "\n  ".join(problems))
    return questions


def question_counts(questions: list[dict]):
    import pandas as pd

    df = pd.DataFrame(questions)
    df["pdf"] = df["document_name"].fillna("(none)").str.slice(0, 30)
    return pd.crosstab(df["group"], df["pdf"], margins=True)


# ── Documents of a run ────────────────────────────────────────────────────────

def _embedding_of(doc: dict) -> str:
    # Entries ingested before the registry recorded embeddings all used the default.
    return (doc.get("embedding") or {}).get("model") or DEFAULT_EMBEDDING


def _matches(doc: dict, run: str) -> bool:
    spec = RUNS[run]
    flags = CONFIGS[spec["config"]]
    if any(bool(doc.get(k)) != v if isinstance(v, bool) else doc.get(k) != v for k, v in flags.items()):
        return False
    # Documents ingested with NER (before it was removed) are not C1–C3 documents.
    if doc.get("ner"):
        return False
    if _embedding_of(doc) != spec.get("embedding", DEFAULT_EMBEDDING):
        return False
    if flags["captioning"] and (doc.get("models") or {}).get("captioning") != spec.get("caption_model"):
        return False
    return True


def resolve_documents(backend: Backend, run: str, questions: list[dict]) -> list[dict]:
    """The run's documents — one per PDF named in the questions.

    Pinned in output/<run>/documents.json the first time, so all repeats of a run use the
    same documents. Otherwise RUNS[run]["document_ids"] if given, else the newest registry
    entry per PDF whose flags, ingestion models and embedding match the run's configuration.
    """
    pinned = _read_json(run_dir(run) / "documents.json")
    if pinned:
        return pinned["documents"]

    registry = sorted(backend.documents(), key=lambda d: d.get("uploaded_at") or "", reverse=True)
    pdfs = sorted({q["document_name"] for q in questions if q.get("document_name")})
    explicit = RUNS[run].get("document_ids")
    chosen, missing = [], []
    for pdf in pdfs:
        candidates = [d for d in registry if d["document_name"] == pdf
                      and (d["document_id"] in explicit if explicit else _matches(d, run))]
        if candidates:
            chosen.append(candidates[0])
        else:
            missing.append(pdf)
    if missing:
        spec = RUNS[run]
        raise RuntimeError(
            f"{run} needs a {spec['config']} document ({CONFIGS[spec['config']]}, caption model "
            f"{spec.get('caption_model')}, embedding {spec.get('embedding', DEFAULT_EMBEDDING)}) for: {missing}. "
            "Ingest them (README), or set RUNS[run]['document_ids'].")
    _write_json(run_dir(run) / "documents.json", {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run": run,
        "spec": RUNS[run],
        "documents": chosen,
    })
    return chosen


# ── Answers ───────────────────────────────────────────────────────────────────

def _strip_text(chunks: list[dict]) -> list[dict]:
    return [{k: v for k, v in c.items() if k != "text"} for c in chunks]


def answer_one(backend: Backend, idx: int, item: dict, document_ids: list[str]) -> dict:
    start = time.perf_counter()
    meta = {k: v for k, v in item.items() if k != "answer"}
    record = {"id": idx, **meta, "expected_answer": item.get("answer"), "expects_refusal": expects_refusal(item)}
    try:
        response = backend.ask(item["question"], document_ids)
        rag = response.get("rag")
        if rag is None:
            raise RuntimeError("response has no `rag` object — restart the backend so it supports include_context")
        record.update(
            intent=rag.get("intent"),
            model=rag.get("model"),
            embedding_model=rag.get("embedding_model"),
            rewritten_query=rag.get("rewritten_query"),
            answer=response["choices"][0]["message"]["content"].strip(),
            citations=response.get("citations") or [],
            chunks=_strip_text(rag.get("chunks") or []),
            candidates=_strip_text(rag.get("candidates") or []),
            contexts=rag.get("contexts") or [],
            conversation_id=response.get("conversation_id"),
            usage=rag.get("usage"),  # per-question model usage (backend rag.usage)
        )
    except Exception as e:  # keep going on per-question failures
        record.update(answer=None, error=f"{type(e).__name__}: {e}")
    record["elapsed_s"] = round(time.perf_counter() - start, 2)
    return record


def _check_backend_models(record: dict, run: str) -> None:
    spec = RUNS[run]
    model, embedding = record.get("model"), record.get("embedding_model")
    if model and model != spec["answer_model"]:
        raise RuntimeError(f"The backend answered with {model!r}; {run} needs {spec['answer_model']!r}. "
                           "Set LLM_PROVIDER / OPENAI_CHAT_MODEL in backend/.env and restart the backend. "
                           "Nothing was saved for this question.")
    want = spec.get("embedding", DEFAULT_EMBEDDING)
    if embedding and embedding != want:
        raise RuntimeError(f"The backend embeds with {embedding!r}; {run} needs {want!r}. Set "
                           "EMBEDDING_PROVIDER / QDRANT_COLLECTION_NAME in backend/.env and restart the backend.")


def pinned_versions() -> dict:
    """What the plan pins per run: the question set (content hash), and the backend's Docling."""
    import hashlib
    import subprocess

    docling = None
    python = BACKEND / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = BACKEND / ".venv" / "bin" / "python"
    try:
        docling = subprocess.run([str(python), "-c", "import importlib.metadata as m; print(m.version('docling'))"],
                                 capture_output=True, text=True, timeout=60).stdout.strip() or None
    except Exception:
        pass
    return {
        "questions_sha256": hashlib.sha256(QUESTIONS_PATH.read_bytes()).hexdigest()[:16],
        "docling_version": docling,
        "openai_temperature": __import__("os").getenv("OPENAI_TEMPERATURE") or "unset (model default)",
    }


def run_answers(backend: Backend, run: str, repeat: int, questions: list[dict] | None = None,
                concurrent: int = 4, redo: bool = False) -> dict:
    """Answer every question once (repeat `repeat`), checkpointing output/<run>/answers-r<repeat>.json.

    The first answer is checked against the run's answering and embedding model before
    any other question is sent, so a misconfigured backend costs one question.
    """
    questions = questions or load_questions()
    documents = resolve_documents(backend, run, questions)
    document_ids = [d["document_id"] for d in documents]
    out_path = run_dir(run) / f"answers-r{repeat}.json"
    previous = {} if redo else _read_json(out_path)
    done = {r["question"]: r for r in previous.get("results", []) if r.get("question") and not r.get("error")}

    by_id: dict[int, dict] = {}
    for i, q in enumerate(questions, 1):
        if q["question"] in done:
            by_id[i] = {**done[q["question"]], "id": i}
    lock = threading.Lock()
    pinned = previous.get("pinned") or pinned_versions()

    def save() -> dict:
        results = [by_id[i] for i in sorted(by_id)]
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "api_url": backend.api_url,
            "run": run,
            "repeat": repeat,
            "spec": RUNS[run],
            "pinned": pinned,
            "documents": documents,
            "scope_document_ids": document_ids,
            "models": sorted({r["model"] for r in results if r.get("model")}),
            "embedding_models": sorted({r["embedding_model"] for r in results if r.get("embedding_model")}),
            "total": len(questions),
            "count": len(results),
            "errors": sum(1 for r in results if r.get("error")),
            "results": results,
        }
        _write_json(out_path, payload)
        return payload

    todo = [(i, q) for i, q in enumerate(questions, 1) if i not in by_id]
    print(f"{run} r{repeat}: answering {len(todo)} questions ({len(by_id)} already done) "
          f"over {len(document_ids)} documents, {concurrent} at a time")
    payload = save()
    if not todo:
        return payload

    def finish(i: int, record: dict) -> None:
        # Every answer, not only the first: a backend restarted mid-run on other settings
        # must stop the run, not mix two models into it.
        _check_backend_models(record, run)
        with lock:
            by_id[i] = record
            save()
        status = f"ERROR {record['error'][:120]}" if record.get("error") else "ok"
        print(f"[{i:>3}] {status} ({record['elapsed_s']}s) {record['question'][:70]}", flush=True)

    # One question alone first: it proves the backend runs the right models.
    first_i, first_q = todo[0]
    record = answer_one(backend, first_i, first_q, document_ids)
    _check_backend_models(record, run)
    finish(first_i, record)

    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {pool.submit(answer_one, backend, i, q, document_ids): i for i, q in todo[1:]}
        for future in as_completed(futures):
            finish(futures[future], future.result())
    payload = save()
    print(f"Done: {payload['count'] - payload['errors']} ok, {payload['errors']} errors -> {out_path}")
    return payload


def load_answers(run: str, repeat: int) -> dict:
    return _read_json(run_dir(run) / f"answers-r{repeat}.json")


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def expected_pages(source: str | None) -> list[int]:
    return sorted({int(p) for p in SOURCE_PAGE.findall(source or "")})


def _relevant(chunk: dict, document_id: str, pages: list[int], tolerance: int) -> bool:
    return (chunk.get("document_id") == document_id and chunk.get("page") is not None
            and any(abs(chunk["page"] - page) <= tolerance for page in pages))


def retrieval_rows(answers: dict, ks=DEFAULT_K, tolerance: str = DEFAULT_TOLERANCE) -> list[dict]:
    """One row per in-scope question with a page source: ranks, hits, correct-PDF rates."""
    ids_by_name = {d["document_name"]: d["document_id"] for d in answers.get("documents", [])}
    rows = []
    for r in answers.get("results", []):
        pages = expected_pages(r.get("source"))
        document_id = ids_by_name.get(r.get("document_name"))
        if r.get("error") or r.get("group") == "out_of_scope" or not pages or not document_id:
            continue
        chunks, candidates = r.get("chunks") or [], r.get("candidates") or []
        first = next((c["rank"] for c in chunks if _relevant(c, document_id, pages, TOLERANCES[tolerance])), None)
        top5 = [c for c in chunks if c.get("rank", 99) <= 5]
        row = {
            "question": r["question"],
            "group": r.get("group"),
            "document_name": r.get("document_name"),
            "reciprocal_rank": 1 / first if first else 0.0,
            "candidates_hit": float(any(_relevant(c, document_id, pages, TOLERANCES[tolerance]) for c in candidates)),
            "correct_pdf@5": float(any(c.get("document_id") == document_id for c in top5)),
            "correct_pdf@1": float(bool(chunks) and chunks[0].get("document_id") == document_id),
            "pdf_precision": (sum(c.get("document_id") == document_id for c in chunks) / len(chunks)) if chunks else 0.0,
        }
        for k in ks:
            row[f"hit@{k}"] = float(first is not None and first <= k)
        rows.append(row)
    return rows


# ── Judge ─────────────────────────────────────────────────────────────────────

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
# metric -> judge model. answer_accuracy is on Sonnet: in the dry run Haiku scored it 0.22
# lower than Sonnet on the same answers, while context_recall agreed on 18 of 20.
JUDGES = {
    "faithfulness": HAIKU,
    "answer_relevancy": HAIKU,
    "context_recall": HAIKU,
    "answer_accuracy": SONNET,
}
CHECK_METRIC, CHECK_JUDGE = "context_recall", SONNET  # check=True adds context_recall_sonnet
REFUSAL_JUDGE = HAIKU
NEEDS_CONTEXT = {"faithfulness", "context_recall"}
# Thinking off: verdicts are short, and Sonnet 5.5 rejects {"type": "disabled"}.
THINKING = {HAIKU: {"type": "disabled"}, SONNET: {"type": "between_tools"}}
PRICES = {HAIKU: (0.10, 0.50), SONNET: (2.00, 10.00)}  # $ per million tokens, input / output
JUDGE_EMBEDDING = "text-embedding-3-small"  # answer_relevancy compares generated questions by embedding
JUDGE_MAX_TOKENS = 4096
JUDGE_TIMEOUT_S = 120.0
JUDGE_MAX_RETRIES = 5

_usage_lock = threading.Lock()
_usage: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})


async def _meter(response) -> None:
    """httpx2 response hook on the Anthropic client: adds each call's usage to _usage."""
    if not response.url.path.endswith("/v1/messages") or response.status_code != 200:
        return
    await response.aread()
    try:
        body = response.json()
    except ValueError:
        return
    model = next((m for m in PRICES if (body.get("model") or "").startswith(m)), body.get("model") or "?")
    u = body.get("usage") or {}
    with _usage_lock:
        _usage[model]["calls"] += 1
        _usage[model]["input_tokens"] += (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                                          + u.get("cache_creation_input_tokens", 0))
        _usage[model]["output_tokens"] += u.get("output_tokens", 0)


def usage_snapshot() -> dict:
    with _usage_lock:
        return {m: dict(u) for m, u in _usage.items()}


def _usage_since(before: dict) -> dict:
    out = {}
    for model, u in usage_snapshot().items():
        b = before.get(model, {})
        delta = {k: u[k] - b.get(k, 0) for k in u}
        p_in, p_out = PRICES.get(model, (0, 0))
        delta["cost_usd"] = round(delta["input_tokens"] / 1e6 * p_in + delta["output_tokens"] / 1e6 * p_out, 4)
        if delta["calls"]:
            out[model] = delta
    return out


def _anthropic_client():
    import httpx2
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(http_client=httpx2.AsyncClient(timeout=JUDGE_TIMEOUT_S, event_hooks={"response": [_meter]}),
                          max_retries=JUDGE_MAX_RETRIES)


def judge_llm(model: str, client=None):
    """A RAGAS LLM on Claude. Haiku/Sonnet 5.5 reject non-default temperature/top_p, and Sonnet
    rejects the forced tool call Instructor uses by default, so the JSON comes back as text."""
    import instructor
    from ragas.llms import llm_factory

    client = client or _anthropic_client()
    llm = llm_factory(model, provider="anthropic", client=client, max_tokens=JUDGE_MAX_TOKENS)
    llm.model_args.pop("temperature", None)
    llm.model_args.pop("top_p", None)
    llm.model_args["thinking"] = THINKING[model]
    llm.client = instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_JSON)
    return llm


def build_metrics(check: bool = False) -> dict:
    """Metric name -> callable(sample) returning a coroutine."""
    import ragas.metrics.collections as C
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory

    client = _anthropic_client()
    llms = {model: judge_llm(model, client) for model in set(JUDGES.values()) | {CHECK_JUDGE}}
    embeddings = embedding_factory(
        "openai", JUDGE_EMBEDDING,
        client=AsyncOpenAI(http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0),
                                                          timeout=httpx.Timeout(30.0, connect=15.0))))

    def make(name: str, llm):
        if name == "faithfulness":
            m = C.Faithfulness(llm=llm)
            return lambda s: m.ascore(user_input=s["question"], response=s["answer"], retrieved_contexts=s["contexts"])
        if name == "answer_relevancy":
            m = C.AnswerRelevancy(llm=llm, embeddings=embeddings)
            return lambda s: m.ascore(user_input=s["question"], response=s["answer"])
        if name == "context_recall":
            m = C.ContextRecall(llm=llm)
            return lambda s: m.ascore(user_input=s["question"], retrieved_contexts=s["contexts"], reference=s["reference"])
        if name == "answer_accuracy":
            m = C.AnswerAccuracy(llm=llm)
            return lambda s: m.ascore(user_input=s["question"], response=s["answer"], reference=s["reference"])
        raise ValueError(name)

    metrics = {name: make(name, llms[model]) for name, model in JUDGES.items()}
    if check:
        metrics[f"{CHECK_METRIC}_sonnet"] = make(CHECK_METRIC, llms[CHECK_JUDGE])
    return metrics


_REFUSAL_PROMPT = """Question: {question}

Answer given by an assistant that may only use a fixed set of documents:
{answer}

Does the answer say that the documents do not contain this information (or otherwise decline to give it), \
rather than giving a specific answer to the question? Reply with exactly YES or NO."""


async def _refused(client, question: str, answer: str) -> float:
    response = await client.messages.create(
        model=REFUSAL_JUDGE, max_tokens=8, thinking=THINKING[REFUSAL_JUDGE],
        messages=[{"role": "user", "content": _REFUSAL_PROMPT.format(question=question, answer=answer)}])
    text = "".join(b.text for b in response.content if b.type == "text").strip().upper()
    if text.startswith("YES"):
        return 1.0
    if text.startswith("NO"):
        return 0.0
    raise ValueError(f"refusal judge replied {text!r}")


def _sample(record: dict) -> dict:
    return {
        "question": record["question"],
        "answer": (record.get("answer") or "").strip(),
        "reference": (record.get("expected_answer") or "").strip(),
        "contexts": [c.strip() for c in record.get("contexts") or [] if c.strip()],
    }


def _wanted(record: dict, metrics: dict) -> list[str]:
    """The metrics that apply to this record."""
    if record.get("expects_refusal"):
        return ["refused"]
    s = _sample(record)
    return [n for n in metrics
            if s["reference"] and (s["contexts"] or n.split("_sonnet")[0] not in NEEDS_CONTEXT)]


async def _score_one(record: dict, names: list[str], metrics: dict, client, semaphore) -> dict:
    """All applicable metrics for one answer; a failing metric is recorded against that metric only."""
    async with semaphore:
        start = time.perf_counter()
        sample = _sample(record)
        scores, errors = {}, {}

        async def run(name):
            try:
                if name == "refused":
                    value = await _refused(client, sample["question"], sample["answer"])
                else:
                    result = await metrics[name](sample)
                    value = getattr(result, "value", result)
                if value is None or math.isnan(float(value)):
                    errors[name] = "judge returned no usable score"
                    return
                scores[name] = round(float(value), 4)
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"[:200]

        await asyncio.gather(*(run(n) for n in names))
        elapsed = round(time.perf_counter() - start, 2)
        shown = " ".join(f"{n[:6]}={scores[n]:.2f}" for n in names if n in scores)
        print(f"[{record['id']:>3}] {'!' + str(len(errors)) if errors else 'ok'} {shown} ({elapsed}s) "
              f"{record['question'][:40]}", flush=True)
        return {"scores": scores, "errors": errors or None, "elapsed_s": elapsed}


def judge_identity() -> dict:
    import ragas

    return {"judges": JUDGES, "check": {CHECK_METRIC: CHECK_JUDGE}, "refusal": REFUSAL_JUDGE,
            "thinking": THINKING, "embedding": JUDGE_EMBEDDING, "ragas_version": ragas.__version__}


async def run_judge(run: str, repeat: int, check: bool = False, concurrent: int = 3, redo: bool = False) -> dict:
    """Score answers-r<repeat>.json into judge-r<repeat>.json; resumes unless the judge setup changed."""
    answers = load_answers(run, repeat)
    if not answers:
        raise RuntimeError(f"No answers for {run} r{repeat} — run the answer step first.")
    out_path = run_dir(run) / f"judge-r{repeat}.json"
    identity = judge_identity()
    previous = {} if redo else _read_json(out_path)
    if previous and {k: v for k, v in previous.get("judge", {}).items() if k != "check"} != \
            {k: v for k, v in identity.items() if k != "check"}:
        print("judge setup changed since the last run — re-scoring everything")
        previous = {}

    metrics = build_metrics(check=check)
    done = {r["question"]: r for r in previous.get("results", []) if not r.get("errors")}
    results, todo, skipped = {}, [], []
    for record in answers["results"]:
        base = {k: record.get(k) for k in ("id", "question", "group", "document_name", "failure_type", "expects_refusal")}
        if record.get("error") or not (record.get("answer") or "").strip():
            skipped.append({**base, "reason": "no answer" if not record.get("error") else record["error"][:100]})
            continue
        names = _wanted(record, metrics)
        old = done.get(record["question"])
        if old and all(n in (old.get("scores") or {}) for n in names):
            results[record["id"]] = {**base, **{k: old[k] for k in ("scores", "errors", "elapsed_s")}}
        else:
            # keep what was already scored; ask only for the missing metrics
            kept = (old or {}).get("scores") or {}
            todo.append((record, base, [n for n in names if n not in kept], kept))

    usage_before = usage_snapshot()
    cost_so_far = previous.get("judge_usage") or {}

    def save() -> dict:
        usage = _usage_since(usage_before)
        for model, u in cost_so_far.items():  # add the earlier sessions' spend
            merged = usage.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            for k in ("calls", "input_tokens", "output_tokens"):
                merged[k] += u.get(k, 0)
            merged["cost_usd"] = round(merged["cost_usd"] + u.get("cost_usd", 0.0), 4)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run": run,
            "repeat": repeat,
            "models": answers.get("models"),
            "judge": identity,
            "judge_usage": usage,
            "count": len(results),
            "skipped": skipped,
            "results": [results[i] for i in sorted(results)],
        }
        _write_json(out_path, payload)
        return payload

    print(f"{run} r{repeat}: judging {len(todo)} answers ({len(results)} already scored, "
          f"{len(skipped)} without an answer), {concurrent} at a time{' + Sonnet check' if check else ''}")
    payload = save()
    if not todo:
        return payload

    client = _anthropic_client()
    semaphore = asyncio.Semaphore(concurrent)

    async def task(record, base, names, kept):
        scored = await _score_one(record, names, metrics, client, semaphore)
        scored["scores"] = {**kept, **scored["scores"]}
        return record["id"], {**base, **scored}

    for finished in asyncio.as_completed([task(*t) for t in todo]):
        rid, row = await finished
        results[rid] = row
        payload = save()  # checkpoint after every answer
    total = sum(u["cost_usd"] for u in payload["judge_usage"].values())
    print(f"Done -> {out_path}  (judge spend for this file so far: ${total:.2f})")
    return payload


# ── Analysis ──────────────────────────────────────────────────────────────────

def run_rows(run: str, repeats=REPEATS):
    """One row per (repeat, question): retrieval metrics and judge scores side by side."""
    import pandas as pd

    frames = []
    for repeat in repeats:
        answers = load_answers(run, repeat)
        if not answers:
            continue
        meta = pd.DataFrame([{k: r.get(k) for k in ("question", "group", "document_name", "expects_refusal")}
                             for r in answers["results"]])
        retrieval = pd.DataFrame(retrieval_rows(answers))
        judged = _read_json(run_dir(run) / f"judge-r{repeat}.json")
        scores = pd.DataFrame([{"question": r["question"], **(r.get("scores") or {})} for r in judged.get("results", [])])
        df = meta
        for extra in (retrieval, scores):
            if not extra.empty:
                df = df.merge(extra.drop(columns=[c for c in ("group", "document_name") if c in extra]),
                              on="question", how="left")
        df.insert(0, "repeat", repeat)
        df.insert(0, "run", run)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


METRICS = ("hit@1", "hit@3", "hit@5", "reciprocal_rank", "correct_pdf@5", "correct_pdf@1", "candidates_hit",
           "faithfulness", "answer_relevancy", "context_recall", "answer_accuracy", "refused")


def summary(runs=tuple(RUNS), by: str | None = "group"):
    """Mean of every metric per run (and group), over all repeats; `n` = questions."""
    import pandas as pd

    df = pd.concat([run_rows(r) for r in runs], ignore_index=True)
    if df.empty:
        return df
    keys = ["run"] + ([by] if by else [])
    columns = [m for m in METRICS if m in df]
    out = df.groupby(keys, sort=False)[columns].mean()
    out.insert(0, "n", df.groupby(keys, sort=False)["question"].nunique())
    return out.round(3)


def paired_comparison(run_a: str, run_b: str, metric: str, groups=IN_SCOPE,
                      n_boot: int = 10_000, seed: int = 0) -> dict:
    """Run B − run A on the questions both scored, each averaged over its repeats.

    Paired bootstrap over questions: resample questions with replacement, take the mean
    difference; the 2.5th / 97.5th percentiles are the 95% CI.
    """
    def per_question(run):
        df = run_rows(run)
        if df.empty or metric not in df:
            return {}
        df = df[df["group"].isin(groups)].dropna(subset=[metric])
        return df.groupby("question")[metric].mean().to_dict()

    a, b = per_question(run_a), per_question(run_b)
    common = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in common]
    row = {"A": run_a, "B": run_b, "metric": metric, "groups": "+".join(groups) if groups != IN_SCOPE else "in-scope",
           "n": len(common)}
    if not diffs:
        return {**row, "A_mean": None, "B_mean": None, "diff": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    low, high = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {
        **row,
        "A_mean": round(sum(a[q] for q in common) / len(common), 3),
        "B_mean": round(sum(b[q] for q in common) / len(common), 3),
        "diff": round(sum(diffs) / len(diffs), 3),
        "ci_low": round(low, 3),
        "ci_high": round(high, 3),
        "up": sum(d > 0 for d in diffs),
        "down": sum(d < 0 for d in diffs),
        "same": sum(d == 0 for d in diffs),
        "verdict": "B better" if low > 0 else "A better" if high < 0 else "no measurable effect",
    }


def comparisons(n_boot: int = 10_000):
    """The plan's comparisons, each with its pre-registered primary metric."""
    import pandas as pd

    return pd.DataFrame([paired_comparison(a, b, m, g, n_boot) for a, b, m, g in COMPARISONS])


def judge_agreement(runs=tuple(RUNS)):
    """Haiku vs Sonnet on context_recall, over every answer scored with check=True."""
    import pandas as pd

    df = pd.concat([run_rows(r) for r in runs], ignore_index=True)
    col = f"{CHECK_METRIC}_sonnet"
    if df.empty or col not in df:
        return {"n": 0}
    pair = df.dropna(subset=[CHECK_METRIC, col])
    diff = (pair[CHECK_METRIC] - pair[col]).abs()
    return {
        "metric": CHECK_METRIC,
        "n": len(pair),
        "haiku_mean": round(float(pair[CHECK_METRIC].mean()), 3),
        "sonnet_mean": round(float(pair[col].mean()), 3),
        "mean_abs_diff": round(float(diff.mean()), 3),
        "identical_share": round(float((diff == 0).mean()), 3),
        "correlation": round(float(pair[CHECK_METRIC].corr(pair[col])), 3) if len(pair) > 2 else None,
    }


def judge_spend(runs=tuple(RUNS)):
    """Anthropic spend recorded in every judge file, per run and model."""
    import pandas as pd

    rows = []
    for run in runs:
        for path in sorted(run_dir(run).glob("judge-r*.json")):
            for model, u in (_read_json(path).get("judge_usage") or {}).items():
                rows.append({"run": run, "file": path.name, "model": model, **u})
    return pd.DataFrame(rows)


# ── Answering cost ────────────────────────────────────────────────────────────

# $ per million tokens (input, output), standard tier as listed on 23 Oct 2026.
ANSWER_PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-5.6-luna": (0.20, 1.20)}
# Qwen is priced by model working time: the seconds llama-server spent on each call, times the GPU's
# hourly rate. RunPod queueing, cold starts and job overhead are recorded but not charged here.
GPU_PRICE_PER_HOUR = 0.69  # RunPod RTX PRO 6000 MIG 24 GB slice, per hour (23 Oct 2026)


def _call_cost(call: dict, gpu_price_per_hour: float) -> float:
    if call.get("provider") == "runpod":
        return (call.get("model_ms") or 0) / 3_600_000 * gpu_price_per_hour
    model = call.get("model") or ""
    price = next((p for name, p in ANSWER_PRICES.items() if model.startswith(name)), None)
    if price is None:
        return float("nan")
    return (call.get("prompt_tokens") or 0) / 1e6 * price[0] + (call.get("completion_tokens") or 0) / 1e6 * price[1]


def answering_cost(runs=tuple(RUNS), gpu_price_per_hour: float = GPU_PRICE_PER_HOUR):
    """Per run, per question (mean over every answer of every repeat): model calls, tokens, time, cost.

    `model_s` is model working time (RunPod: llama-server time); `wall_s` is the question's end-to-end
    time as the evaluation saw it (median). OpenAI query embeddings (~30 tokens) are not counted.
    """
    import statistics

    import pandas as pd

    rows = []
    for run in runs:
        answers = [r for repeat in REPEATS for r in (load_answers(run, repeat).get("results") or [])
                   if r.get("answer") and isinstance(r.get("usage"), dict) and "by_call" in r["usage"]]
        if not answers:
            continue
        n = len(answers)
        calls = [c for r in answers for c in r["usage"]["by_call"]]
        total = lambda key: sum(c.get(key) or 0 for c in calls)  # noqa: E731
        cost = sum(_call_cost(c, gpu_price_per_hour) for c in calls)
        rows.append({
            "run": run,
            "answers": n,
            "calls/q": round(len(calls) / n, 2),
            "prompt_tokens/q": round(total("prompt_tokens") / n),
            "completion_tokens/q": round(total("completion_tokens") / n),
            "model_s/q": round(total("model_ms") / n / 1000, 2) if total("model_ms") else None,
            "runpod_exec_s/q": round(total("execution_ms") / n / 1000, 2) if total("execution_ms") else None,
            "runpod_delay_s/q": round(total("delay_ms") / n / 1000, 2) if total("delay_ms") else None,
            "wall_s/q (median)": round(statistics.median(r.get("elapsed_s") or 0 for r in answers), 1),
            "cost per 1,000 questions ($)": round(cost / n * 1000, 2),
        })
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()
