"""RAGAS scoring for an answers-*.json run (IMKA RAG output).

Scores a run produced by run-questions-imka.py — which records, per question,
the assembled `context`, the `answer`, and the `expected_answer` from questions.json
— with the RAGAS metric set, and writes test/output/ragas-<run>.json.

Pick the judge(s) explicitly — at least one of --luna / --qwen is required, and
passing both scores every question twice, once per judge:

  --luna   primary judge: gpt-5.6-luna on OpenAI. ~$0.35 for 70 questions.
  --qwen   secondary judge: the app's own Qwen3.8-27B on RunPod. Free but slow, and
           it grades answers its own model produced — a judge-agreement check, not
           an independent verdict. Sends each question's independent calls at once
           and keeps --max-jobs (default 18, ~6 per RunPod worker) jobs in flight,
           earliest question first.

Both judges write into ONE file, `test/output/ragas-<run>.json`, with per-question
scores nested per judge plus a flat `rows` array ready for charting. Running one
judge never disturbs the other's scores.

Built to compare several IMKA versions over time while the question set keeps
changing, so:

- It scores *any* answers file (`-i`), not just one hardcoded run.
- Already-scored questions are skipped per judge, matched on the exact question
  text, so reordering questions.json or adding questions to it does not invalidate
  the scores already paid for. `--redo` re-scores the selected judge(s).
- Each judge's model/provider is recorded next to its scores. Change a judge's
  model and only that judge's scores go stale.
- The output is rewritten after every scored question, so a crash or a Ctrl-C
  never loses judge tokens already spent.

Metrics (ragas.metrics.collections):
  faithfulness         is the answer grounded in the retrieved context (hallucination)
  answer_relevancy     does the answer address the question               (+ embeddings)
  context_precision    were the retrieved chunks the ones that mattered   (vs reference)
  context_recall       is the reference content present in the context    (retrieval miss)
  answer_accuracy      answer vs expected_answer — the headline accuracy number.
                       Deliberately NOT ragas's FactualCorrectness: that metric
                       computes its true positives by decomposing the *response*
                       into claims and checking each against the *reference*, in
                       every mode (precision/recall/f1 only change the denominator).
                       The references in questions.json are terse values ("360 Nm")
                       while the answers cite pages and add context, so almost no
                       response claim is entailed by the reference, tp is 0, and it
                       returns 0.00 for demonstrably correct answers. AnswerAccuracy
                       sees the question too and scores those same answers 1.00,
                       including the absence probes where refusing IS the right answer.
  semantic_similarity  embedding-only, no judge. Read it with care: cosine between a
                       paragraph answer and a 6-character reference is low even when
                       the answer is right. A rough signal, not a verdict.

Usage (from repo root, using the dedicated eval venv — NOT backend/.venv):
  python -m venv test\\.venv
  test\\.venv\\Scripts\\python.exe -m pip install -r test\\requirements-eval.txt

  test\\.venv\\Scripts\\python.exe test\\evaluate-ragas.py --luna --dry-run
  test\\.venv\\Scripts\\python.exe test\\evaluate-ragas.py --luna --limit 3
  test\\.venv\\Scripts\\python.exe test\\evaluate-ragas.py --luna
  test\\.venv\\Scripts\\python.exe test\\evaluate-ragas.py --qwen
  test\\.venv\\Scripts\\python.exe test\\evaluate-ragas.py --luna --qwen
  test\\.venv\\Scripts\\python.exe test\\evaluate-ragas.py --luna -i test\\output\\answers-imka-v2.json
"""
import argparse
import asyncio
import contextvars
import heapq
import itertools
import json
import math
import os
import statistics
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "backend"

# pip-system-certs globally injects pip's vendored truststore into `ssl`; openai's
# httpx then wraps it in truststore again, recursing forever on TLS handshake
# (surfaces as "Connection error."). Undo the injection.
try:
    from pip._vendor import truststore as _pip_truststore

    _pip_truststore.extract_from_ssl()
except Exception:
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND / ".env")

import httpx  # noqa: E402
import ragas  # noqa: E402
import ragas.metrics.collections as C  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402

# How rag/pipeline.py:_stage4_assemble_context joins the reranked chunks. Splitting
# on it recovers the individual chunks RAGAS needs as `retrieved_contexts`; the
# "[Document Name, p. N]" label is deliberately left on each chunk (the judge uses it).
# If that join ever changes, this must change with it.
CONTEXT_SEPARATOR = "\n\n---\n\n"

DEFAULT_JUDGE = "gpt-5.6-luna"

# The selectable judges. `key` is what appears in the output JSON and on --flags.
JUDGE_KEYS = ("luna", "qwen")

# Judge token budget per call. RAGAS defaults to 1024, which truncates structured
# output on reasoning models and surfaces as a parse failure rather than a short answer.
JUDGE_MAX_TOKENS = 4096

# The openai client defaults to a 600s timeout, so a connection silently dropped
# mid-flight (a TLS-inspecting proxy reaping idle sockets — the same environment
# that makes pip-system-certs necessary here) stalls a worker slot for ten minutes
# with no output. Observed in practice: all 4 slots blocked on Established sockets,
# zero CPU, no progress for 6+ minutes. Fail fast and retry instead.
JUDGE_TIMEOUT_S = 120.0
JUDGE_CONNECT_TIMEOUT_S = 15.0
JUDGE_MAX_RETRIES = 3  # bounded: worst case ~3 x 120s on one call, not one 600s stall

# Embeddings connections do not survive reuse here: the next request on a kept-alive
# /embeddings connection either fails instantly or — worse — gets no reply at all and
# sits out the full timeout. Logged in practice: from question 3 on, every question
# lost exactly 120s to one such embeddings call (131s/question instead of ~11s).
# Chat completions reuse connections fine. So embeddings open a fresh connection per
# call (a TLS handshake, ~0.3s) and get a short timeout — they normally answer in <1s.
EMBEDDING_TIMEOUT_S = 30.0


def no_reuse_http_client() -> httpx.AsyncClient:
    """An httpx client that never keeps a connection alive for a later request."""
    return httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0),
                             timeout=httpx.Timeout(EMBEDDING_TIMEOUT_S, connect=JUDGE_CONNECT_TIMEOUT_S))


def judge_client() -> AsyncOpenAI:
    """An OpenAI client that gives up on a stalled request instead of hanging."""
    return AsyncOpenAI(
        timeout=httpx.Timeout(JUDGE_TIMEOUT_S, connect=JUDGE_CONNECT_TIMEOUT_S),
        max_retries=JUDGE_MAX_RETRIES,
    )


def embeddings_client() -> AsyncOpenAI:
    """An OpenAI client for /embeddings: fresh connection per call, short timeout."""
    return AsyncOpenAI(http_client=no_reuse_http_client(), max_retries=JUDGE_MAX_RETRIES)

# Questions whose expected answer is "the manual does not say this". context_recall
# and factual_correctness are not meaningful against a "nothing to find" reference,
# so these are flagged in the output and grouped separately in the aggregate rather
# than being silently averaged in with the rest.
ABSENCE_FAILURE_TYPES = ("Absence probe", "fabrication probe", "Deferred-value")

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_accuracy",
    "semantic_similarity",
)

# Progress-line labels only — context_precision/context_recall would otherwise
# both truncate to "cont".
SHORT_LABELS = {
    "faithfulness": "faith",
    "answer_relevancy": "relev",
    "context_precision": "cprec",
    "context_recall": "crec",
    "answer_accuracy": "acc",
    "semantic_similarity": "sim",
}


def is_absence_question(failure_type: str | None) -> bool:
    return any(marker in (failure_type or "") for marker in ABSENCE_FAILURE_TYPES)


# ── Judge client ──────────────────────────────────────────────────────────────

def is_reasoning_model(model: str) -> bool:
    """Does this judge need the reasoning-model parameter shape?

    RAGAS asks the same question in `InstructorLLM._map_openai_params`, but its
    parser reads "gpt-5.6-luna" as version "5.6", cannot int() it, and concludes
    no — so it sends `max_tokens` and the request 400s. This handles the dotted
    minor version (gpt-5.6) that the current model names actually use.
    """
    model = model.lower()
    if len(model) >= 2 and model[0] == "o" and model[1].isdigit():
        return True  # o1, o3, o4-mini, ...
    if model.startswith("gpt-"):
        try:
            return float(model[4:].split("-")[0]) >= 5  # gpt-5, gpt-5.6-luna, ...
        except ValueError:
            return False
    return model == "codex-mini"


def apply_judge_params(llm, judge_model: str) -> list[str]:
    """Reshape `llm.model_args` into what the judge model accepts.

    RAGAS defaults to temperature=0.01, top_p=0.1, max_tokens=N. Reasoning models
    reject all three: they want `max_completion_tokens`, accept only temperature
    1.0, and do not take top_p at all.
    """
    if not is_reasoning_model(judge_model):
        return []
    args, changes = llm.model_args, []
    if "max_tokens" in args:
        args["max_completion_tokens"] = args.pop("max_tokens")
        changes.append("max_tokens -> max_completion_tokens")
    if args.get("temperature") != 1.0:
        args["temperature"] = 1.0
        changes.append("temperature -> 1.0")
    if args.pop("top_p", None) is not None:
        changes.append("dropped top_p")
    return changes


# ── RunPod job scheduling (Qwen judge) ────────────────────────────────────────

# The question a RunPod call belongs to. Set in score_one; asyncio copies it into every
# task a question spawns, so each call knows its question without threading it through ragas.
QUESTION_PRIORITY: contextvars.ContextVar[int] = contextvars.ContextVar("question_priority", default=0)

# One question sends 12 RunPod jobs at once (see build_runpod_metrics), then 1 more.
QWEN_JOBS_PER_QUESTION = 12
DEFAULT_MAX_JOBS = 18


class JobLimiter:
    """Caps the RunPod jobs in flight; when full, the earliest question's call goes next.

    Keeps a steady queue on the endpoint so workers never wait for work, without
    flooding it past RUNPOD_REQUEST_TIMEOUT. Because the waiter from the lowest
    question id always wins, a started question (e.g. its faithfulness step 2)
    finishes before later questions take its slots.
    """

    def __init__(self, max_jobs: int):
        self.max_jobs = max_jobs
        self.in_flight = 0
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._order = itertools.count()

    async def acquire(self, priority: int) -> None:
        if self.in_flight < self.max_jobs and not self._waiters:
            self.in_flight += 1
            return
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._order), future))
        try:
            await future  # release() hands its slot straight to this waiter
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()  # the slot arrived just as we were cancelled — pass it on
            raise

    def release(self) -> None:
        while self._waiters:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                future.set_result(None)
                return
        self.in_flight -= 1


def build_runpod_metrics(names: list[str], embedding_model: str, max_jobs: int) -> tuple[dict, object]:
    """Same six metrics, judged by the app's own Qwen3.8-27B on RunPod.

    Uses RAGAS's *legacy* metric API, not `ragas.metrics.collections`: the collections
    metrics need an `InstructorBaseRagasLLM`, i.e. an OpenAI-style client with native
    structured output, and the RunPod endpoint has no tool/function calling at all
    (see backend/rag/llm_client.py). The legacy metrics ask for JSON in the prompt and
    parse it out of the text reply, which ChatRunPod can do. They are deprecated but
    still present in ragas 0.4.3 and compute the same quantities.

    Embeddings stay on OpenAI either way — they are cheap and not part of what a
    judge comparison is testing.

    Scheduled for throughput: every call independent of another is sent at once, so a
    question submits 12 RunPod jobs together (faithfulness step 1, relevancy x3,
    precision x5, recall, accuracy x2) and only faithfulness step 2 follows. A shared
    JobLimiter keeps at most `max_jobs` in flight, earliest question first. Scores are
    unchanged — same prompts, same parsing, same aggregation; only the timing differs.
    """
    sys.path.insert(0, str(BACKEND))
    import numpy as np
    from langchain_core.prompt_values import StringPromptValue
    from langchain_openai import OpenAIEmbeddings
    # ChatRunPod directly, NOT get_chat_model(): that follows the backend's LLM_PROVIDER,
    # which defaults to openai — the "qwen" judge would silently be gpt-4o-mini.
    from rag.llm_client import ChatRunPod
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._context_precision import QAC, Verification
    from ragas.metrics.base import ensembler

    limiter = JobLimiter(max_jobs)

    class QueuedChatRunPod(ChatRunPod):
        """ChatRunPod that waits for a JobLimiter slot before submitting to RunPod."""

        async def _acall(self, messages, stop, kwargs):
            await limiter.acquire(QUESTION_PRIORITY.get())
            try:
                return await super()._acall(messages, stop, kwargs)
            finally:
                limiter.release()

    # The legacy API is deprecated in favour of ragas.metrics.collections, which we
    # cannot use here (no structured output on RunPod — see the docstring above).
    # The warning is expected; silence it so the progress output stays readable.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (AnswerAccuracy, Faithfulness,
                                   LLMContextPrecisionWithReference, LLMContextRecall,
                                   ResponseRelevancy, SemanticSimilarity)

        # The wrappers warn when constructed, not just when imported.
        chat = QueuedChatRunPod(temperature=0.0)
        llm = LangchainLLMWrapper(chat)
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=embedding_model, timeout=EMBEDDING_TIMEOUT_S, max_retries=JUDGE_MAX_RETRIES,
            http_async_client=no_reuse_http_client()))

    class ParallelContextPrecision(LLMContextPrecisionWithReference):
        """ragas's context precision, judging all chunks at once instead of one by one.

        Each chunk's verdict is independent; results keep chunk order, which the
        average-precision formula depends on.
        """

        async def _ascore(self, row, callbacks):
            user_input, retrieved_contexts, reference = self._get_row_attributes(row)
            responses = await asyncio.gather(*(
                self.context_precision_prompt.generate_multiple(
                    data=QAC(question=user_input, context=context, answer=reference),
                    llm=self.llm, callbacks=callbacks)
                for context in retrieved_contexts))
            answers = [Verification(**ensembler.from_discrete(
                [[result.model_dump() for result in verdicts]], "verdict")[0])
                for verdicts in responses]
            return self._calculate_average_precision(answers)

    class ParallelAnswerAccuracy(AnswerAccuracy):
        """ragas's answer accuracy, running its two independent judge prompts at once."""

        async def _single_turn_ascore(self, sample, callbacks):
            async def judge(template, answer0, answer1, inference, truth):
                score = np.nan
                for _ in range(self.retry):  # retry when no rating appears in the reply
                    prompt = StringPromptValue(text=template.format(
                        query=sample.user_input, answer0=answer0, answer1=answer1,
                        sentence_inference=inference, sentence_true=truth))
                    reply = await self.llm.agenerate_text(prompt, n=1, temperature=0.10)
                    score = self.process_score(reply.generations[0][0].text)
                    if score == score:  # not NaN
                        break
                return score

            try:
                score_ref_gen, score_gen_ref = await asyncio.gather(
                    judge(self.template_accuracy1, "User Answer", "Reference Answer",
                          sample.response, sample.reference),
                    judge(self.template_accuracy2, "Reference Answer", "User Answer",
                          sample.reference, sample.response))
            except Exception:
                return np.nan  # same as ragas: a failed sample scores NaN
            return self.average_scores(score_ref_gen, score_gen_ref)

    legacy = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ParallelContextPrecision(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
        "answer_accuracy": ParallelAnswerAccuracy(llm=llm),
        "semantic_similarity": SemanticSimilarity(embeddings=embeddings),
    }

    def scorer(metric):
        # Legacy metrics take one SingleTurnSample and return a bare float.
        return lambda s: metric.single_turn_ascore(SingleTurnSample(
            user_input=s["question"], response=s["answer"],
            retrieved_contexts=s["contexts"], reference=s["reference"]))

    unknown = set(names) - set(legacy)
    if unknown:
        raise SystemExit(f"unknown metric(s): {', '.join(sorted(unknown))}\n"
                         f"available: {', '.join(legacy)}")
    return {name: scorer(legacy[name]) for name in names}, chat


def build_metrics(names: list[str], judge_model: str, embedding_model: str) -> tuple[dict, object]:
    """(metric name -> callable taking the sample dict and returning a coroutine, llm)."""
    llm = llm_factory(judge_model, provider="openai", client=judge_client(),
                      max_tokens=JUDGE_MAX_TOKENS)
    embeddings = embedding_factory("openai", embedding_model, client=embeddings_client())

    faithfulness = C.Faithfulness(llm=llm)
    relevancy = C.AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = C.ContextPrecisionWithReference(llm=llm)
    recall = C.ContextRecall(llm=llm)
    accuracy = C.AnswerAccuracy(llm=llm)
    similarity = C.SemanticSimilarity(embeddings=embeddings)

    available = {
        "faithfulness": lambda s: faithfulness.ascore(
            user_input=s["question"], response=s["answer"], retrieved_contexts=s["contexts"]),
        "answer_relevancy": lambda s: relevancy.ascore(
            user_input=s["question"], response=s["answer"]),
        "context_precision": lambda s: precision.ascore(
            user_input=s["question"], reference=s["reference"], retrieved_contexts=s["contexts"]),
        "context_recall": lambda s: recall.ascore(
            user_input=s["question"], retrieved_contexts=s["contexts"], reference=s["reference"]),
        "answer_accuracy": lambda s: accuracy.ascore(
            user_input=s["question"], response=s["answer"], reference=s["reference"]),
        "semantic_similarity": lambda s: similarity.ascore(
            reference=s["reference"], response=s["answer"]),
    }
    unknown = set(names) - set(available)
    if unknown:
        raise SystemExit(f"unknown metric(s): {', '.join(sorted(unknown))}\n"
                         f"available: {', '.join(available)}")
    return {name: available[name] for name in names}, llm


# ── Loading ───────────────────────────────────────────────────────────────────

def load_answers(path: Path) -> tuple[dict, list[dict]]:
    """The run's header fields and its per-question records."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, payload.get("results", [])


def to_sample(record: dict) -> tuple[dict | None, str | None]:
    """A scoreable sample, or (None, reason) for a record RAGAS cannot score."""
    if record.get("error"):
        return None, f"run error: {record['error'][:120]}"
    answer = (record.get("answer") or "").strip()
    if not answer:
        return None, "no answer"
    reference = (record.get("expected_answer") or "").strip()
    if not reference:
        return None, "no expected_answer in questions.json"
    contexts = [c.strip() for c in (record.get("context") or "").split(CONTEXT_SEPARATOR) if c.strip()]
    if not contexts:
        return None, "no retrieved context (run with context, not --no-context)"
    return {"question": record["question"], "answer": answer,
            "reference": reference, "contexts": contexts}, None


JUDGE_IDENTITY = ("model", "provider", "embedding_model", "ragas_version")


def load_previous(out_path: Path) -> dict:
    """The existing combined file, or an empty shell if there is none / it is corrupt."""
    if not out_path.exists():
        return {}
    try:
        return json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  !! {out_path.name} is not valid JSON — ignoring it and scoring everything")
        return {}


def reusable_scores(previous: dict, records: list[dict], judge_key: str, judge: dict,
                    metrics: list[str]) -> dict[int, dict]:
    """What this judge already scored, keyed by index in the *current* answers file.

    Matched on question text, so scores survive questions.json being reordered or
    extended. An entry is dropped — and therefore re-scored — when that metric run
    errored, when a requested metric is missing, or when this judge's identity
    (model/provider/embeddings/ragas version) has changed since.
    """
    previous_judge = (previous.get("judges") or {}).get(judge_key, {})
    if previous_judge:
        changed = [k for k in JUDGE_IDENTITY if previous_judge.get(k) != judge[k]]
        if changed:
            print(f"  !! {judge_key} judge changed ({', '.join(changed)}) — re-scoring it")
            return {}

    by_question = {}
    for r in previous.get("results", []):
        entry = (r.get("scores") or {}).get(judge_key)
        if not r.get("question") or not entry:
            continue
        if (r.get("errors") or {}).get(judge_key):
            continue
        if any(isinstance(v, float) and math.isnan(v) for v in entry.values()):
            continue  # written by an older run that stored NaN scores
        if all(m in entry for m in metrics):
            by_question[r["question"]] = entry

    return {i: by_question[r["question"]]
            for i, r in enumerate(records, 1)
            if r.get("question") in by_question}


# ── Scoring ───────────────────────────────────────────────────────────────────

async def score_one(idx: int, judge_key: str, question: str, sample: dict, metrics: dict,
                    semaphore: asyncio.Semaphore) -> dict:
    """Score one question with one judge.

    A failing metric is recorded against that metric only — it does not void the
    question, and the other judge's scores are untouched.
    """
    async with semaphore:
        QUESTION_PRIORITY.set(idx)  # inherited by every RunPod call this question makes
        start = time.perf_counter()
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}

        async def run(name, make_coro):
            try:
                metric_result = await make_coro(sample)
                value = getattr(metric_result, "value", metric_result)
                if value is not None and math.isnan(float(value)):
                    # ragas scores a failed/unparseable judgement as NaN; record it as
                    # an error so it's excluded from means and re-scored next run.
                    errors[name] = "judge returned no usable score (NaN)"
                    return
                scores[name] = None if value is None else round(float(value), 4)
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"[:200]

        await asyncio.gather(*(run(name, fn) for name, fn in metrics.items()))

        elapsed = round(time.perf_counter() - start, 2)
        shown = " ".join(f"{SHORT_LABELS.get(n, n[:4])}={scores[n]:.2f}"
                         for n in metrics if scores.get(n) is not None)
        tag = f"!{len(errors)}" if errors else "ok"
        print(f"[{judge_key}][{idx:>3}] {tag} {shown} ({elapsed}s) {question[:42]}", flush=True)
        return {
            "id": idx,
            "scores": {name: scores.get(name) for name in metrics},
            "errors": errors or None,
            "elapsed_s": elapsed,
        }


# ── Aggregation ───────────────────────────────────────────────────────────────

def judge_scores(record: dict, judge_key: str) -> dict:
    return (record.get("scores") or {}).get(judge_key) or {}


def summarise(results: list[dict], metrics: list[str], judge_key: str) -> dict:
    """Mean and n per metric, ignoring questions where that metric has no score."""
    summary = {}
    for name in metrics:
        values = [judge_scores(r, judge_key)[name] for r in results
                  if judge_scores(r, judge_key).get(name) is not None]
        summary[name] = {
            "mean": round(statistics.mean(values), 4) if values else None,
            "n": len(values),
        }
    return summary


def aggregate(results: list[dict], metrics: list[str], judge_key: str) -> dict:
    """Overall means plus the breakdowns this question set was designed around."""
    scored = [r for r in results if judge_scores(r, judge_key)]

    def grouped(key):
        groups: dict[str, list[dict]] = {}
        for r in scored:
            groups.setdefault(r.get(key) or "(none)", []).append(r)
        return {k: {**summarise(v, metrics, judge_key), "questions": len(v)}
                for k, v in sorted(groups.items())}

    graded = [r for r in scored if not r.get("reference_is_absence")]
    absence = [r for r in scored if r.get("reference_is_absence")]
    return {
        "overall": {**summarise(scored, metrics, judge_key), "questions": len(scored)},
        # context_recall / answer_accuracy against a "not in the manual" reference
        # behave differently — this is the number to compare across IMKA versions.
        "excluding_absence_probes": {**summarise(graded, metrics, judge_key),
                                     "questions": len(graded)},
        "absence_probes_only": {**summarise(absence, metrics, judge_key),
                                "questions": len(absence)},
        "by_failure_type": grouped("failure_type"),
        "by_rag_target": grouped("rag_target"),
    }


def build_rows(results: list[dict], metrics: list[str], judge_keys: list[str]) -> list[dict]:
    """Long/tidy format: one row per (question, judge, metric).

    This is the shape plotting libraries want — group by `judge` for a judge
    comparison, by `rag_target` for the per-failure-mode breakdown, pivot on
    `metric` for a per-metric bar chart. Saves every consumer re-flattening the
    nested structure.
    """
    rows = []
    for r in results:
        for judge_key in judge_keys:
            scores = judge_scores(r, judge_key)
            for name in metrics:
                if scores.get(name) is None:
                    continue
                rows.append({
                    "question_id": r["id"],
                    "judge": judge_key,
                    "metric": name,
                    "score": scores[name],
                    "failure_type": r.get("failure_type"),
                    "rag_target": r.get("rag_target"),
                    "reference_is_absence": r.get("reference_is_absence"),
                })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def default_out(input_path: Path) -> Path:
    """answers-<run>.json -> ragas-<run>.json. One file holds every judge's scores."""
    name = input_path.name
    stem = name[len("answers-"):] if name.startswith("answers-") else name
    stem = stem[:-len(".json")] if stem.endswith(".json") else stem
    return input_path.parent / f"ragas-{stem}.json"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=Path, default=HERE / "output" / "answers-imka.json",
                        help="answers-*.json produced by a run-questions-* script")
    parser.add_argument("-o", "--out", type=Path, help="default: the input name with answers- -> ragas-")
    parser.add_argument("--luna", action="store_true",
                        help=f"judge with {DEFAULT_JUDGE} on OpenAI — fast, parallel, ~$0.35/70 questions")
    parser.add_argument("--qwen", action="store_true",
                        help="judge with the app's own Qwen3.8-27B on RunPod — free but slow, "
                             "and it grades answers its own model produced")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE,
                        help=f"override the model --luna uses (default {DEFAULT_JUDGE})")
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--metrics", default=",".join(METRIC_NAMES),
                        help=f"comma-separated subset of: {', '.join(METRIC_NAMES)}")
    parser.add_argument("--concurrent", type=int,
                        help="questions scored in parallel (default: 1 for --luna; for --qwen, "
                             "enough to keep --max-jobs RunPod jobs queued)")
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS,
                        help=f"--qwen only: RunPod jobs in flight at once, earliest question first "
                             f"(default {DEFAULT_MAX_JOBS}; ~6 per RunPod worker keeps workers busy)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--redo", action="store_true",
                        help="re-score the selected judge(s), ignoring their existing scores")
    parser.add_argument("--dry-run", action="store_true", help="report what would be scored, no API calls")
    args = parser.parse_args()

    selected = [key for key in JUDGE_KEYS if getattr(args, key)]
    if not selected:
        parser.error("pick at least one judge: --luna and/or --qwen")

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    judges = {
        "luna": {
            "model": args.judge_model,
            "provider": "openai",
            "embedding_model": args.embedding_model,
            "ragas_version": ragas.__version__,
            "metric_api": "collections",
            "metrics": metric_names,
        },
        "qwen": {
            "model": os.getenv("RUNPOD_MODEL_ALIAS", "qwen3.8-27b-q4"),
            "provider": "runpod",
            "embedding_model": args.embedding_model,
            "ragas_version": ragas.__version__,
            "metric_api": "legacy",
            "metrics": metric_names,
        },
    }
    # Luna: one question at a time by default. Running 4 in parallel kept every worker
    # slot blocked on a dropped socket at once (see JUDGE_TIMEOUT_S); sequential holds
    # one connection instead of four and is far easier to reason about when something
    # stalls. Raise it with --concurrent if the network behaves.
    # Qwen: the JobLimiter caps RunPod load, so admit enough questions to keep
    # --max-jobs queued — one extra so the next question's burst is ready to go.
    concurrency = {
        "luna": args.concurrent or 1,
        "qwen": args.concurrent or math.ceil(args.max_jobs / QWEN_JOBS_PER_QUESTION) + 1,
    }

    out_path = args.out or default_out(args.input)
    run_header, records = load_answers(args.input)
    # --limit caps what gets *scored*, never what gets kept: the output file is
    # rewritten whole every checkpoint, so dropping records here would delete
    # scores from earlier full runs.
    scoreable_ids = set(range(1, args.limit + 1)) if args.limit else None

    samples: dict[int, dict] = {}
    skipped: list[dict] = []
    for i, record in enumerate(records, 1):
        sample, reason = to_sample(record)
        if sample is None:
            skipped.append({"id": i, "question": record.get("question"), "reason": reason})
        else:
            samples[i] = sample

    if args.dry_run:
        chunks = sum(len(s["contexts"]) for s in samples.values())
        context_chars = sum(len(c) for s in samples.values() for c in s["contexts"])
        print(f"input      {args.input}")
        print(f"out        {out_path}")
        print(f"run        {run_header.get('mode')} / {run_header.get('model')} "
              f"({run_header.get('generated_at')})")
        print(f"scoreable  {len(samples)} questions, {chunks} context chunks "
              f"({context_chars:,} chars ~= {context_chars // 4:,} tokens)")
        if skipped:
            print(f"skipped    {len(skipped)}")
            for s in skipped[:10]:
                print(f"           [{s['id']}] {s['reason']}")
        print(f"metrics    {', '.join(metric_names)}")
        for key in selected:
            j = judges[key]
            print(f"judge      {key}: {j['model']} via {j['provider']} / {j['embedding_model']} "
                  f"({j['metric_api']} API), concurrency {concurrency[key]}"
                  + (f", max {args.max_jobs} RunPod jobs" if key == "qwen" else ""))
        # Each context-using metric reads the whole context at least once, and
        # faithfulness/precision/recall each take more than one pass over it.
        context_tokens = context_chars // 4
        print(f"\nRough judge input: ~{context_tokens:,} tokens per pass over the contexts; "
              f"the context metrics take several passes each, so order "
              f"{context_tokens * 8 // 1000:,}k-{context_tokens * 15 // 1000:,}k input tokens "
              f"per judge.")
        print("No API calls made.")
        return

    previous = load_previous(out_path)

    # Every question that can be scored, carrying its static fields. Scores from a
    # previous run are merged in below, per judge.
    by_id: dict[int, dict] = {}
    for i, record in enumerate(records, 1):
        if i not in samples:
            continue
        by_id[i] = {
            "id": i,
            "question": record["question"],
            "failure_type": record.get("failure_type"),
            "rag_target": record.get("rag_target"),
            "source": record.get("source"),
            "expected_answer": record.get("expected_answer"),
            "reference_is_absence": is_absence_question(record.get("failure_type")),
            "n_contexts": len(samples[i]["contexts"]),
            "scores": {},
            "errors": {},
            "elapsed_s": {},
        }

    # Carry every stored score forward, for both judges. Scores this run is about to
    # replace are cleared per judge below; everything else — the other judge, and any
    # question outside --limit — must survive, because save() rewrites the whole file.
    stored_judges = dict(previous.get("judges") or {})
    previous_by_question = {r["question"]: r for r in previous.get("results", [])
                            if r.get("question")}
    for record in by_id.values():
        prior = previous_by_question.get(record["question"], {})
        for key, entry in (prior.get("scores") or {}).items():
            record["scores"][key] = entry
            if (prior.get("errors") or {}).get(key):
                record["errors"][key] = prior["errors"][key]
            if (prior.get("elapsed_s") or {}).get(key):
                record["elapsed_s"][key] = prior["elapsed_s"][key]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save() -> dict:
        """Rewrite the combined scores file; returns the payload.

        Written to a temp file and renamed, so a crash mid-write cannot leave a
        half-finished (unparseable) JSON file behind.
        """
        results = [by_id[i] for i in sorted(by_id)]
        present = [k for k in JUDGE_KEYS
                   if any(judge_scores(r, k) for r in results)]
        judge_blocks = {}
        for key in present:
            meta = judges[key] if key in selected else stored_judges.get(key, {})
            scored = [r for r in results if judge_scores(r, key)]
            judge_blocks[key] = {
                **meta,
                "count": len(scored),
                "errors": sum(1 for r in scored if (r.get("errors") or {}).get(key)),
            }
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scored_file": str(args.input),
            "run": {k: run_header.get(k) for k in
                    ("mode", "model", "endpoint_id", "generated_at", "source_file")},
            "questions_total": len(records),
            "judges": judge_blocks,
            "metrics": metric_names,
            "skipped": skipped,
            "aggregate": {k: aggregate(results, metric_names, k) for k in present},
            "results": results,
            "rows": build_rows(results, metric_names, present),
        }
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(out_path)
        return payload

    payload = save()
    for key in selected:
        judge = judges[key]
        reusable = {} if args.redo else reusable_scores(previous, records, key, judge, metric_names)
        # Questions in scope for this run: everything, or just the --limit head.
        targets = [i for i in by_id if scoreable_ids is None or i in scoreable_ids]
        for i in targets:
            if i in reusable:
                by_id[i]["scores"][key] = reusable[i]
            else:  # stale, errored, or --redo — drop it so it gets scored again
                by_id[i]["scores"].pop(key, None)
                by_id[i]["errors"].pop(key, None)
                by_id[i]["elapsed_s"].pop(key, None)
        todo = [i for i in targets if key not in by_id[i]["scores"]]
        done = {i: by_id[i]["scores"][key] for i in targets if key in by_id[i]["scores"]}

        print(f"\n[{key}] scoring {len(todo)} questions "
              f"({len(done)} already scored, {len(skipped)} unscoreable) "
              f"from {args.input.name}")
        print(f"  {judge['model']} via {judge['provider']}, embeddings {judge['embedding_model']}, "
              f"ragas {ragas.__version__}, concurrency {concurrency[key]}"
              + (f", max {args.max_jobs} RunPod jobs" if key == "qwen" else ""))

        if not todo:
            payload = save()
            continue

        if key == "qwen":
            metrics, _ = build_runpod_metrics(metric_names, args.embedding_model, args.max_jobs)
            print("  note: Qwen is grading answers its own model produced — expect self-bias, "
                  "and expect this to take hours.")
        else:
            metrics, llm = build_metrics(metric_names, judge["model"], args.embedding_model)
            adaptations = apply_judge_params(llm, judge["model"])
            if adaptations:
                print(f"  judge params: {'; '.join(adaptations)}")

        semaphore = asyncio.Semaphore(concurrency[key])
        pending = [asyncio.create_task(
            score_one(i, key, by_id[i]["question"], samples[i], metrics, semaphore))
            for i in todo]
        for task in asyncio.as_completed(pending):
            result = await task
            record = by_id[result["id"]]
            record["scores"][key] = result["scores"]
            record["elapsed_s"][key] = result["elapsed_s"]
            if result["errors"]:
                record["errors"][key] = result["errors"]
            payload = save()  # checkpoint after every question

    print(f"\nDone. Means excluding absence probes:")
    header = "  " + " " * 22 + "".join(f"{k:>12}" for k in payload["judges"])
    print(header)
    for name in metric_names:
        cells = ""
        for key in payload["judges"]:
            mean = payload["aggregate"][key]["excluding_absence_probes"][name]["mean"]
            cells += f"{'n/a' if mean is None else f'{mean:.3f}':>12}"
        print(f"  {name:22}{cells}")
    for key, block in payload["judges"].items():
        print(f"  {key}: {block['count']} scored, {block['errors']} with metric errors")
    print(f"  {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
