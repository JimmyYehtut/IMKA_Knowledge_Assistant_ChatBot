"""IMKA RAG run over questions.json, through the running backend API.

Sends each question from questions.json to the live backend exactly like the chat
UI does — POST /v1/chat/completions on http://localhost:8000 — so the answer comes
from whatever LLM, embeddings and Qdrant collection that backend is running with:
intent classification -> query rewrite -> retrieve -> BM25 rerank -> context
assembly -> answer -> citations. The script knows nothing about the model; switch
the backend's model and rerun with -o to compare.

The request sets the backend's non-standard `include_context: true`, so the
response also carries the intent, rewritten query and assembled context that
evaluate-ragas.py and evaluate-retrieval.py score. Every question is saved as a
conversation in the backend's chat history, like any other chat.

Writes test/output/answers-imka.json (override with -o): answer, intent, rewritten
query, citations, optional context, plus the expected answer/source from
questions.json; rewritten after every question so a crash or a Ctrl-C never loses
the answers already produced.

Questions already answered successfully in the output file are skipped —
matched on the exact question text, so reordering or inserting questions in
questions.json does not invalidate existing answers. Pass --redo to ignore the
existing file and answer everything again.

Needs the backend and Qdrant running (cd backend && docker compose up -d &&
uvicorn api.main:app --port 8000) and an IMKA user account. Credentials come
from --email/--password or IMKA_EMAIL/IMKA_PASSWORD (environment or test/.env);
the API address from --api-url or IMKA_API_URL (default http://localhost:8000).
Only needs httpx + python-dotenv, so it runs on test/.venv.

Usage (from the test folder, test/.venv active):
  python run-questions-imka.py
  python run-questions-imka.py --limit 2
  python run-questions-imka.py --no-context
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_API_URL = "http://localhost:8000"
# One answer can take minutes (RunPod cold start, a long context), so be generous.
DEFAULT_TIMEOUT_S = 900.0


def load_questions(path: Path) -> list[dict]:
    """questions.json (list of objects) or questions.txt (one question per line)."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [{**q, "question": q["question"].strip()} for q in data]
    lines = path.read_text(encoding="utf-8").splitlines()
    return [{"question": q.strip()} for q in lines if q.strip() and not q.strip().startswith("#")]


def load_done(out_path: Path, questions: list[dict]) -> dict[int, dict]:
    """Previously answered questions, keyed by their index in the *current* questions file.

    Matched on question text (not id), so answers survive questions.json being
    reordered or extended. Failed attempts are dropped so they get retried.
    """
    if not out_path.exists():
        return {}
    try:
        prev = json.loads(out_path.read_text(encoding="utf-8")).get("results", [])
    except json.JSONDecodeError:
        print(f"  !! {out_path.name} is not valid JSON — ignoring it and answering everything")
        return {}
    by_question = {r["question"]: r for r in prev if r.get("question") and not r.get("error")}
    return {
        i: {**by_question[q["question"]], "id": i}
        for i, q in enumerate(questions, 1)
        if q["question"] in by_question
    }


class Backend:
    """Minimal client for the IMKA backend API: login + chat completions."""

    def __init__(self, api_url: str, email: str, password: str, timeout: float):
        self.api_url = api_url.rstrip("/")
        self.email, self.password = email, password
        self.http = httpx.Client(base_url=self.api_url, timeout=httpx.Timeout(timeout, connect=10.0))
        self.token: str | None = None

    def check(self) -> None:
        """Fail fast with a readable message when the backend is not up."""
        try:
            self.http.get("/v1/models").raise_for_status()
        except httpx.HTTPError as e:
            raise SystemExit(f"Backend not reachable at {self.api_url} ({type(e).__name__}: {e}).\n"
                             "Start it: cd backend && uvicorn api.main:app --port 8000")

    def login(self) -> None:
        r = self.http.post("/api/auth/login", json={"email": self.email, "password": self.password})
        if r.status_code == 401:
            raise SystemExit(f"Login failed for {self.email}: invalid credentials. "
                             "Use an account that exists in the IMKA UI.")
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def ask(self, question: str) -> dict:
        body = {"messages": [{"role": "user", "content": question}], "include_context": True}
        for attempt in (1, 2):
            r = self.http.post("/v1/chat/completions", json=body,
                               headers={"Authorization": f"Bearer {self.token}"})
            if r.status_code == 401 and attempt == 1:  # token expired mid-run — log in again once
                self.login()
                continue
            if r.is_error:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()


def answer_one(backend: Backend, idx: int, item: dict, include_context: bool) -> dict:
    start = time.perf_counter()
    question = item["question"]
    # carry the question's grading metadata through; `answer` is the expected answer
    meta = {k: v for k, v in item.items() if k != "answer"}
    result = {"id": idx, **meta, "expected_answer": item.get("answer")}
    try:
        response = backend.ask(question)
        rag = response.get("rag")
        if rag is None:
            raise RuntimeError("response has no `rag` object — the backend is older than this script "
                               "(restart it so it supports include_context)")
        result.update(
            intent=rag.get("intent"),
            rewritten_query=rag.get("rewritten_query"),
            answer=response["choices"][0]["message"]["content"].strip(),
            citations=response.get("citations") or [],
            conversation_id=response.get("conversation_id"),
        )
        if include_context:
            result["context"] = rag.get("context") or ""
    except Exception as e:  # keep going on per-question failures
        result.update(answer=None, error=f"{type(e).__name__}: {e}")
    result["elapsed_s"] = round(time.perf_counter() - start, 2)
    status = "ERROR" if "error" in result else "ok"
    note = "" if result.get("intent") in (None, "knowledge") else f" [intent={result['intent']}, no retrieval]"
    print(f"[{idx:>3}] {status} ({result['elapsed_s']}s) {question[:70]}{note}", flush=True)
    if "error" in result:
        print(f"      {result['error'][:200]}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=Path, default=HERE / "questions.json")
    parser.add_argument("-o", "--out", type=Path, default=HERE / "output" / "answers-imka.json")
    parser.add_argument("--api-url", default=os.getenv("IMKA_API_URL", DEFAULT_API_URL))
    parser.add_argument("--email", default=os.getenv("IMKA_EMAIL"))
    parser.add_argument("--password", default=os.getenv("IMKA_PASSWORD"))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per question")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--no-context", action="store_true", help="omit retrieved context from the JSON")
    parser.add_argument("--redo", action="store_true", help="ignore existing answers and answer every question again")
    args = parser.parse_args()

    if not args.email or not args.password:
        parser.error("IMKA credentials missing: pass --email/--password or set IMKA_EMAIL/IMKA_PASSWORD "
                     "(environment or test/.env)")

    questions = load_questions(args.input)[: args.limit]
    out_path = args.out

    done = {} if args.redo else load_done(out_path, questions)
    todo = [(i, q) for i, q in enumerate(questions, 1) if i not in done]

    backend = Backend(args.api_url, args.email, args.password, args.timeout)
    backend.check()
    backend.login()
    print(
        f"Answering {len(todo)} questions ({len(done)} already in {out_path.name}) "
        f"from {args.input.name} via {backend.api_url} as {args.email}, one at a time"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(by_id: dict[int, dict]) -> int:
        """Rewrite the output file from what is answered so far; returns the error count.

        Written to a temp file and renamed, so a crash mid-write cannot leave a
        half-finished (unparseable) JSON file behind.
        """
        results = [by_id[i] for i in sorted(by_id)]
        errors = sum(1 for r in results if r.get("error"))
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "rag",
            "api_url": backend.api_url,
            "source_file": str(args.input),
            "total": len(questions),
            "count": len(results),
            "errors": errors,
            "results": results,
        }
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        return errors

    # Strictly one question at a time — don't assume the backend's LLM handles concurrent load.
    by_id = dict(done)
    errors = save(by_id)
    for i, q in todo:
        by_id[i] = answer_one(backend, i, q, not args.no_context)
        errors = save(by_id)  # checkpoint after every question

    not_knowledge = sum(1 for r in by_id.values() if r.get("intent") not in (None, "knowledge"))
    print(f"\nDone: {len(by_id) - errors} ok, {errors} errors"
          + (f", {not_knowledge} not classified as knowledge (no retrieval)" if not_knowledge else ""))
    print(f"  {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted — answers so far are saved; rerun to continue.")
