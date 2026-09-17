"""Baseline 2: whole PDF + Qwen3.8-27B on RunPod, no RAG.

Uses the same client the app uses for generation (`backend/rag/llm_client.py`,
`ChatRunPod`), so this baseline and the RAG run in `run-questions-imka.py`
are answered by the *same* model — the only difference is how the model gets the
document.

The PDF itself goes to RunPod, base64-encoded in the job's `pdf` field together
with `pdf_dpi` (default 120). The worker (`llamacpp-runpod-single` handler, v16+)
renders every page to PNG at that DPI and puts the pages, each after a `[p. N]`
marker, in front of the question.

One question per request, like the Luna and IMKA runs, so answers cannot bleed
into each other and the three runs stay comparable. Every request has the same
prefix (system prompt + a user turn holding the pages) and only the question —
in its own, second user turn — differs, so llama-server's prompt cache
(`cache_prompt`) re-reads just the question once the first request has paid for
the pages (~211k tokens, ~11 min at 120 DPI). The separate turn is what makes
that work on this hybrid model; see build_messages(). The
worker also keeps its last render, so repeat requests skip rendering. Each
answer's `prompt_cache` hit rate is saved and printed — near 100% means the cache
is working; 0% on every question means each one is paying the full prefill.

  !! The FIRST request (and any after the worker was scaled down) takes 11+ min:
  the RunPod endpoint's execution timeout must allow that. --timeout covers the
  handler and this client.

RunPod caps a /run request body at 10 MB. The 5.6 MB ABB manual is ~7.45 MB as
base64, so it fits; the size is checked before anything is sent.

Writes test/output/answers-qwen.json (override with -o), rewritten after every
question so a failed job or a Ctrl-C never loses the answers already produced.
A failed question is recorded with its error and the run carries on; after
--max-consecutive-errors failures in a row (endpoint down, timeouts) it stops
instead of burning time on the rest. Questions already answered successfully are
skipped on the next run — matched on the exact question text, so reordering or
inserting questions in questions.json does not invalidate existing answers.
Pass --redo to answer everything again.

Usage (from repo root, using the backend venv):
  backend\\.venv\\Scripts\\python.exe test\\run-questions-qwen.py --dry-run  # payload size only
  backend\\.venv\\Scripts\\python.exe test\\run-questions-qwen.py --limit 2
  backend\\.venv\\Scripts\\python.exe test\\run-questions-qwen.py
"""
import argparse
import asyncio
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# pip-system-certs globally injects pip's vendored truststore into `ssl`; httpx then
# wraps it in truststore again, recursing forever on TLS handshake (surfaces as a
# generic connection error). Undo the injection.
try:
    from pip._vendor import truststore as _pip_truststore

    _pip_truststore.extract_from_ssl()
except Exception:
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND / ".env")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from rag.llm_client import ChatRunPod, _extract_text  # noqa: E402

RUNPOD_RUN_LIMIT_BYTES = 10_000_000  # /run request body cap (10 MB), decimal to stay conservative

# Identical for every request — it is part of the cached prefix, so keep it stable.
INSTRUCTIONS = (
    "You are a technical assistant answering questions about a manual that is provided as page "
    "images. Each page image is preceded by its page marker [p. N]. "
    "Answer ONLY using that manual. If it does not contain the answer, say so clearly. "
    "Mention the page number(s) you used. Keep answers concise; use Markdown lists/tables when "
    "helpful and LaTeX ($...$ / $$...$$) for formulas."
)

# First user turn — the worker prepends the page images to it. Also part of the cached prefix.
DOCUMENT_TURN = "The pages above are the complete manual. My question follows in the next message."


class ChatRunPodPdf(ChatRunPod):
    """ChatRunPod that attaches a PDF for the worker to render into page images.

    The PDF lives on model fields rather than invoke kwargs so megabytes of base64
    never end up in LangSmith invocation params. `_acall` is re-implemented only to
    keep the handler's whole `output` (usage, finish_reason, cache and render stats)
    in `last_output` — the base client returns the text alone.
    """

    pdf_b64: str = ""
    pdf_dpi: int = 120
    last_output: dict | None = None
    last_job: dict | None = None  # RunPod envelope: which worker ran it, queue/cold-start delay

    def _payload(self, messages, stop, kwargs) -> dict:
        body = super()._payload(messages, stop, kwargs)
        if self.pdf_b64:
            body["input"]["pdf"] = self.pdf_b64
            body["input"]["pdf_dpi"] = self.pdf_dpi
        return body

    async def _acall(self, messages, stop, kwargs) -> str:
        self.last_output = self.last_job = None  # never report the previous question's stats for a failed one
        url = f"{self.base_url}/{self.endpoint_id}"
        headers = self._headers()
        deadline = time.monotonic() + self.request_timeout
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(f"{url}/run", headers=headers, json=self._payload(messages, stop, kwargs))
            response.raise_for_status()
            payload = response.json()
            while self._is_pending(payload):
                if time.monotonic() > deadline:
                    raise self._timeout_error(payload)
                await asyncio.sleep(self.poll_interval)
                status = await client.get(f"{url}/status/{payload['id']}", headers=headers)
                status.raise_for_status()
                payload = status.json()
        # The prompt cache lives in one worker's llama-server, so a miss on a different
        # worker_id (or a large delay_ms, i.e. a cold start) is not a cache bug.
        self.last_job = {
            "job_id": payload.get("id"),
            "worker_id": payload.get("workerId"),
            "delay_ms": payload.get("delayTime"),
            "execution_ms": payload.get("executionTime"),
        }
        self.last_output = payload.get("output") if isinstance(payload.get("output"), dict) else None
        return _extract_text(payload)


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


def build_messages(question: str) -> list:
    """System prompt, then the manual and the question as two SEPARATE user turns.

    The worker attaches the page images to the first user turn. Qwen3.8 is a hybrid
    (linear-attention) model, so llama-server cannot resume from an arbitrary cached
    position — only from a context checkpoint, and it saves one "before the latest
    user message" (llama.cpp PR #22929). With pages and question in one turn that
    checkpoint sits before the pages and every question re-reads all ~211k tokens;
    with the question in its own turn it sits right after the pages.
    """
    return [
        SystemMessage(content=INSTRUCTIONS),
        HumanMessage(content=DOCUMENT_TURN),
        HumanMessage(content=f"Question: {question}"),
    ]


def response_stats(output: dict | None) -> dict:
    """The parts of the handler output worth keeping next to the answer."""
    if not output:
        return {}
    completion = output.get("response") or {}
    choices = completion.get("choices") or [{}]
    return {
        "finish_reason": choices[0].get("finish_reason"),
        "usage": completion.get("usage"),
        "timings": completion.get("timings"),
        "prompt_cache": output.get("prompt_cache"),
        "document": output.get("document"),
        "warnings": output.get("warnings"),
    }


async def ask(llm: ChatRunPodPdf, idx: int, item: dict) -> dict:
    start = time.perf_counter()
    # carry the question's grading metadata through; `answer` is the expected answer
    meta = {k: v for k, v in item.items() if k != "answer"}
    result = {"id": idx, **meta, "expected_answer": item.get("answer")}
    try:
        reply = await llm.ainvoke(
            build_messages(item["question"]),
            config={"run_name": "no_rag_pdf_pages", "metadata": {"source": "no_rag", "question_id": idx}},
        )
        answer = reply.content.strip()
        if answer:
            result["answer"] = answer
        else:
            result.update(answer=None, error="empty answer")
    except Exception as e:  # keep going on per-question failures
        result.update(answer=None, error=f"{type(e).__name__}: {e}")
    result["elapsed_s"] = round(time.perf_counter() - start, 2)
    stats = response_stats(llm.last_output)
    if stats.get("finish_reason") == "length":
        result["truncated"] = True  # hit --max-tokens; the answer is kept but may be cut off
    result.update(stats)
    if llm.last_job:
        result["runpod"] = llm.last_job

    cache = stats.get("prompt_cache") or {}
    usage = stats.get("usage") or {}
    job = llm.last_job or {}
    status = "ERROR" if result.get("error") else "ok"
    tokens = (
        f", prompt {usage.get('prompt_tokens') or 0:,} tok, cache hit {cache.get('hit_rate') or 0:.0%}"
        f", worker {job.get('worker_id')}, delay {(job.get('delay_ms') or 0) / 1000:.0f}s"
        if usage
        else ""
    )
    print(f"[{idx:>3}] {status} ({result['elapsed_s']}s{tokens}) {item['question'][:60]}", flush=True)
    if result.get("error"):
        print(f"      {result['error'][:200]}", flush=True)
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=ROOT / "pdfs" / "ABB_Manual_for_Induction_Motors_and_Generators_EN.pdf")
    parser.add_argument("-i", "--input", type=Path, default=HERE / "questions.json")
    parser.add_argument("-o", "--out", type=Path, default=HERE / "output" / "answers-qwen.json")
    parser.add_argument("--dpi", type=int, default=120, help="page render resolution on the worker (36-300)")
    parser.add_argument("--limit-bytes", type=int, default=RUNPOD_RUN_LIMIT_BYTES, help="RunPod /run body cap")
    parser.add_argument("--max-tokens", type=int, default=2048, help="answer length cap per question")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per question, for both the handler and the client")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=3, help="stop after this many failures in a row")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--redo", action="store_true", help="ignore existing answers and answer every question again")
    parser.add_argument("--dry-run", action="store_true", help="print the request size and exit")
    args = parser.parse_args()

    questions = load_questions(args.input)[: args.limit]
    out_path = args.out
    done = {} if args.redo else load_done(out_path, questions)
    todo = [(i, q) for i, q in enumerate(questions, 1) if i not in done]
    print(f"questions: {len(questions)} from {args.input.name}, {len(done)} already in {out_path.name}, {len(todo)} to answer")
    if not todo and not args.dry_run:
        return

    llm = ChatRunPodPdf(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        handler_timeout=args.timeout,
        request_timeout=args.timeout,
        pdf_b64=base64.b64encode(args.pdf.read_bytes()).decode("ascii"),
        pdf_dpi=args.dpi,
    )

    # ── size check: the largest request must fit RunPod's /run body cap ──────────
    longest = max((q["question"] for q in questions), key=len)
    request_bytes = len(json.dumps(llm._payload(build_messages(longest), None, {})).encode("utf-8"))
    fits = request_bytes <= args.limit_bytes
    print(
        f"{args.pdf.name}: {args.pdf.stat().st_size / 1e6:.2f} MB PDF -> request {request_bytes / 1e6:.2f} MB "
        f"[{'fits' if fits else 'OVER LIMIT'}, limit {args.limit_bytes / 1e6:.0f} MB], rendered on the worker at {args.dpi} DPI"
    )
    if args.dry_run:
        return
    if not fits:
        sys.exit("The request is over the RunPod /run limit — nothing was sent.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(by_id: dict[int, dict]) -> int:
        """Rewrite answers-qwen.json from what is answered so far; returns the error count.

        Written to a temp file and renamed, so a crash mid-write cannot leave a
        half-finished (unparseable) JSON file behind.
        """
        results = [by_id[i] for i in sorted(by_id)]
        errors = sum(1 for r in results if r.get("error"))
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "no-rag-pdf-pages",
            "model": llm.model_alias,
            "endpoint_id": llm.endpoint_id,
            "pdf": args.pdf.name,
            "dpi": args.dpi,
            "request_bytes": request_bytes,
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

    print(
        f"Answering {len(todo)} questions one at a time with {llm.model_alias} on RunPod {llm.endpoint_id} "
        f"(timeout {args.timeout}s each; the first request pays for all page tokens)",
        flush=True,
    )

    # Strictly one question at a time — the prompt cache lives in a single worker's slot.
    by_id = dict(done)
    errors = save(by_id)
    consecutive = 0
    for i, q in todo:
        by_id[i] = await ask(llm, i, q)
        errors = save(by_id)  # checkpoint after every question
        consecutive = consecutive + 1 if by_id[i].get("error") else 0
        if consecutive >= args.max_consecutive_errors:
            print(f"\n  !! {consecutive} failures in a row — stopping. Rerun to retry the failed and remaining questions.")
            break

    print(f"\nDone: {len(by_id) - errors} ok, {errors} errors, {len(questions) - len(by_id)} not attempted")
    print(f"  {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
