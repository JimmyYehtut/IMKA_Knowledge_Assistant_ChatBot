"""Per-request record of what each model call cost, for evaluation traces.

`collect()` starts a record for the current request; every chat or embedding call made while
answering it appends one entry: token counts for OpenAI models (priced per token), and for the
RunPod Qwen endpoint the time llama-server spent on the call (priced per GPU-second — the
endpoint's queue, cold start and job overhead are recorded separately, not counted as model
time). `summary()` totals them.

The record lives in a ContextVar holding a list, so it follows the request into worker threads
started with asyncio.to_thread (which copies the context; the list is shared, not copied).
Outside a request with a record, calls record nothing.
"""
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_records: ContextVar[list[dict] | None] = ContextVar("usage_records", default=None)


def collect() -> list[dict]:
    """Start recording for the current request; returns the (live) list of entries."""
    records: list[dict] = []
    _records.set(records)
    return records


def record(entry: dict) -> None:
    records = _records.get()
    if records is not None:
        records.append(entry)


def record_runpod(job_input: dict, envelope: dict, output: dict) -> None:
    """One RunPod job: llama-server's own time (model working time) plus the endpoint overhead."""
    if _records.get() is None:
        return
    response = output.get("response") or {}
    timings = response.get("timings") or {}
    usage = response.get("usage") or output.get("usage") or {}
    is_embed = "embed" in job_input
    model_ms = (timings.get("prompt_ms") or 0) + (timings.get("predicted_ms") or 0)
    record({
        "provider": "runpod",
        "kind": "embedding" if is_embed else "chat",
        "model": output.get("model") or response.get("model"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        # Chat: llama-server's prompt + generation time. Embedding: the handler's call to the
        # embedding server (llama-server reports no timings for embeddings).
        "model_ms": round(model_ms if not is_embed else (output.get("latency_ms") or 0), 1),
        "handler_ms": output.get("latency_ms"),          # all llama-server calls in the job
        "execution_ms": envelope.get("executionTime"),   # RunPod: time inside the worker
        "delay_ms": envelope.get("delayTime"),           # RunPod: queue + cold start
    })


class OpenAIUsageCallback(BaseCallbackHandler):
    """Records the token usage of every ChatOpenAI call (attached in rag.llm.chat_llm)."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if _records.get() is None:
            return
        # The message's usage_metadata is set for invoked and streamed calls alike (streaming
        # needs stream_usage=True, set in chat_llm); llm_output only for invoked ones.
        message = getattr(response.generations[0][0], "message", None) if response.generations else None
        meta = getattr(message, "usage_metadata", None) or {}
        output = response.llm_output or {}
        if meta:
            prompt, completion = meta.get("input_tokens", 0), meta.get("output_tokens", 0)
            reasoning = (meta.get("output_token_details") or {}).get("reasoning", 0)
        else:
            usage = output.get("token_usage") or {}
            prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        model = output.get("model_name") or (getattr(message, "response_metadata", None) or {}).get("model_name")
        record({
            "provider": "openai",
            "kind": "chat",
            "model": model,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning or 0,
        })


def summary(records: list[dict]) -> dict:
    """Totals for one request: calls, tokens, and (RunPod) model / execution / delay time."""
    totals: dict[str, Any] = {"calls": len(records)}
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "model_ms", "execution_ms", "delay_ms"):
        values = [r.get(key) or 0 for r in records]
        if any(values):
            totals[key] = round(sum(values), 1)
    totals["by_call"] = records
    return totals
