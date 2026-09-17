"""Baseline 1: whole PDF + GPT-5.6-luna (OpenAI Responses API), no RAG.

Uploads the PDF once to the OpenAI Files API, then asks each question from
questions.json with the full PDF attached (`input_file`). The file is placed first
in every request so OpenAI prompt caching can reuse the prefix (cached input is
~10x cheaper).

Writes test/output/answers-luna.json (override with -o): answers + per-question token usage and cost,
rewritten after every answer so a crash or Ctrl-C never loses finished work.

Questions already answered successfully in answers-luna.json are skipped —
matched on the exact question text, so reordering or inserting questions in
questions.json does not invalidate existing answers. Pass --redo to ignore the
existing file and answer everything again.

Each request is ~239k input tokens; with a 500k tokens/min org limit only ~2
requests fit per minute, so 429s are retried with backoff and concurrency is low.

Usage (from repo root, using the backend venv for openai + dotenv):
  backend\\.venv\\Scripts\\python.exe test\\run-questions-luna.py --dry-run  # estimate cost
  backend\\.venv\\Scripts\\python.exe test\\run-questions-luna.py --limit 2
  backend\\.venv\\Scripts\\python.exe test\\run-questions-luna.py
"""
import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

# pip-system-certs globally injects pip's vendored truststore into `ssl`; openai's
# httpx then wraps it in truststore again, recursing forever on TLS handshake
# (surfaces as "Connection error."). Undo the injection.
try:
    from pip._vendor import truststore as _pip_truststore

    _pip_truststore.extract_from_ssl()
except Exception:
    pass

from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI, RateLimitError  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / "backend" / ".env")

MODEL = "gpt-5.6-luna"

# USD per 1M tokens (https://developers.openai.com/api/docs/pricing, standard tier, short context)
PRICING = {
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-sol": {"input": 4.00, "cached_input": 0.40, "output": 20.00},
}

INSTRUCTIONS = (
    "You are a technical assistant answering questions about the attached manual. "
    "Answer ONLY using the attached PDF. If the document does not contain the answer, say so clearly. "
    "Mention the page number(s) you used. Keep answers concise; use Markdown lists/tables when helpful "
    "and LaTeX ($...$ / $$...$$) for formulas."
)


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


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    p = PRICING[model]
    uncached = input_tokens - cached_tokens
    return (uncached * p["input"] + cached_tokens * p["cached_input"] + output_tokens * p["output"]) / 1e6


def build_input(file_id: str, detail: str, question: str) -> list[dict]:
    # File first, question last -> identical prefix across requests -> prompt cache hits.
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id, "detail": detail},
                {"type": "input_text", "text": f"Question: {question}"},
            ],
        }
    ]


async def create_with_retry(client, kwargs: dict, idx: int, attempts: int = 8):
    """Retry 429s: one ~239k-token PDF request uses about half of a 500k tokens/min limit."""
    for attempt in range(attempts):
        try:
            return await client.responses.create(**kwargs)
        except RateLimitError as e:
            if attempt == attempts - 1:
                raise
            m = re.search(r"try again in ([\d.]+)(ms|s)", str(e))
            wait = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) if m else 30.0
            wait = max(wait, 5.0) + random.uniform(0, 5)
            print(f"[{idx:>3}] rate limited, retrying in {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)


async def ask(client, args, file_id: str, idx: int, item: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        start = time.perf_counter()
        question = item["question"]
        # carry the question's grading metadata through; `answer` is the expected answer
        meta = {k: v for k, v in item.items() if k != "answer"}
        result = {"id": idx, **meta, "expected_answer": item.get("answer")}
        try:
            kwargs = dict(
                model=args.model,
                instructions=INSTRUCTIONS,
                input=build_input(file_id, args.detail, question),
                # max 64 chars
                prompt_cache_key="no-rag-" + hashlib.sha256(f"{file_id}:{args.detail}".encode()).hexdigest()[:32],
            )
            if args.reasoning:
                kwargs["reasoning"] = {"effort": args.reasoning}
            resp = await create_with_retry(client, kwargs, idx)
            u = resp.usage
            cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
            reasoning = getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0
            result.update(
                answer=resp.output_text.strip(),
                usage={
                    "input_tokens": u.input_tokens,
                    "cached_input_tokens": cached,
                    "output_tokens": u.output_tokens,
                    "reasoning_tokens": reasoning,
                },
                cost_usd=round(cost_usd(args.model, u.input_tokens, cached, u.output_tokens), 5),
            )
        except Exception as e:  # keep going on per-question failures
            result.update(answer=None, error=f"{type(e).__name__}: {e}")
        result["elapsed_s"] = round(time.perf_counter() - start, 2)
        tag = "ERROR" if "error" in result else f"ok ${result['cost_usd']:.4f}"
        print(f"[{idx:>3}] {tag} ({result['elapsed_s']}s) {question[:60]}", flush=True)
        return result


async def get_file_id(client, pdf: Path) -> str:
    """Upload the PDF once and remember its file_id next to this script."""
    cache_path = HERE / ".openai_file_ids.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = f"{pdf.resolve()}|{pdf.stat().st_size}"
    if key in cache:
        try:
            await client.files.retrieve(cache[key])
            return cache[key]
        except Exception:
            pass
    with pdf.open("rb") as fh:
        f = await client.files.create(file=fh, purpose="user_data")
    cache[key] = f.id
    cache_path.write_text(json.dumps(cache, indent=2))
    return f.id


async def estimate(client, args, file_id: str, questions: list[dict]) -> None:
    count = await client.responses.input_tokens.count(
        model=args.model,
        instructions=INSTRUCTIONS,
        input=build_input(file_id, args.detail, questions[0]["question"]),
    )
    n, per_q = len(questions), count.input_tokens
    p = PRICING[args.model]
    out_guess = 1500  # answer + reasoning tokens per question (rough)
    no_cache = n * (per_q * p["input"] + out_guess * p["output"]) / 1e6
    with_cache = (per_q * p["input"] + (n - 1) * per_q * p["cached_input"] + n * out_guess * p["output"]) / 1e6
    print(f"model={args.model} detail={args.detail} questions={n}")
    print(f"input tokens per question: {per_q:,}  (total {n * per_q:,})")
    # Observed: PDF input_file requests got 0 cached tokens, so expect the no-cache figure.
    print(f"estimated cost: ${no_cache:.2f} without cache hits (expected), ${with_cache:.2f} if prompt caching applies")
    print(f"(assumes ~{out_guess} output+reasoning tokens per answer)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL, choices=sorted(PRICING))
    parser.add_argument("--pdf", type=Path, default=ROOT / "pdfs" / "ABB_Manual_for_Induction_Motors_and_Generators_EN.pdf")
    parser.add_argument("-i", "--input", type=Path, default=HERE / "questions.json")
    parser.add_argument("-o", "--out", type=Path, default=HERE / "output" / "answers-luna.json")
    parser.add_argument("--detail", default="high", choices=["low", "high"], help="PDF page-image detail")
    parser.add_argument("--reasoning", choices=["none", "low", "medium", "high"], help="reasoning effort (model default if unset)")
    parser.add_argument("--concurrent", type=int, default=2, help="keep low: each request is ~239k tokens")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--redo", action="store_true", help="ignore existing answers and answer every question again")
    parser.add_argument("--dry-run", action="store_true", help="count tokens and estimate cost, no answers")
    args = parser.parse_args()

    questions = load_questions(args.input)[: args.limit]
    client = AsyncOpenAI()
    file_id = await get_file_id(client, args.pdf)

    if args.dry_run:
        await estimate(client, args, file_id, questions)
        return

    out_path = args.out
    done = {} if args.redo else load_done(out_path, questions)
    todo = [(i, q) for i, q in enumerate(questions, 1) if i not in done]

    print(f"Answering {len(todo)} questions ({len(done)} already in {out_path.name}) with {args.model} (detail={args.detail}, file={file_id})")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(by_id: dict[int, dict]) -> dict:
        """Rewrite answers-luna.json from what is answered so far; returns the payload.

        Written to a temp file and renamed, so a crash mid-write cannot leave a
        half-finished (unparseable) JSON file behind.
        """
        results = [by_id[i] for i in sorted(by_id)]
        ok = [r for r in results if not r.get("error")]
        totals = {
            k: sum(r["usage"][k] for r in ok)
            for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
        }
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "no-rag",
            "model": args.model,
            "pdf": args.pdf.name,
            "detail": args.detail,
            "reasoning": args.reasoning,
            "source_file": str(args.input),
            "total": len(questions),
            "count": len(results),
            "errors": len(results) - len(ok),
            "usage_total": totals,
            "cost_usd_total": round(sum(r["cost_usd"] for r in ok), 4),
            "results": results,
        }
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        return payload

    by_id = dict(done)
    payload = save(by_id)
    sem = asyncio.Semaphore(args.concurrent)
    if todo:
        # Run the first question alone so the PDF prefix is cached before the rest fan out.
        first = await ask(client, args, file_id, *todo[0], sem)
        by_id[first["id"]] = first
        payload = save(by_id)
        pending = [asyncio.create_task(ask(client, args, file_id, i, q, sem)) for i, q in todo[1:]]
        for task in asyncio.as_completed(pending):
            r = await task
            by_id[r["id"]] = r
            payload = save(by_id)  # checkpoint after every answer

    print(f"\nDone: {payload['count'] - payload['errors']} ok, {payload['errors']} errors, total ${payload['cost_usd_total']:.4f}")
    print(f"tokens: {payload['usage_total']}")
    print(f"  {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
