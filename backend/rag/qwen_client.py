"""Qwen3.8-27B on the RunPod serverless llama.cpp endpoint, as a LangChain chat model.

Selected with LLM_PROVIDER="qwen" (rag/llm.py chat_llm()), so intent detection, query
rewrite, answer generation and topic labelling all run on it. Figure
captioning uses it when DOCLING_CAPTION_MODEL="qwen" (ingest/embed_and_store_docling.py).

Configured via .env:

    RUNPOD_API_KEY=...                 # sent as Bearer
    RUNPOD_ENDPOINT_ID=...             # https://api.runpod.ai/v2/<id>
    QWEN_MODEL_ALIAS=...               # optional; the name reported for runs

The endpoint is not OpenAI-wire-compatible: a job is one `{"input": ...}` object sent to
POST /run and collected by polling GET /status/{id}; the reply text is `output.text`. The
handler catches its own failures and returns COMPLETED with an `error` key, so the status
alone cannot tell success from failure. It forwards no `stop`, no `response_format` and
does not stream, so:

- structured output is done in the prompt: the JSON schema is appended to the system
  turn and the reply is parsed and validated (one retry with the validation error);
- `.astream()` yields the whole answer as one chunk (LangChain's fallback).

Sampling defaults are Qwen's recommended non-thinking settings; thinking is off.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Sequence

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

from rag import usage as _usage

DEFAULT_MODEL_ALIAS = "Qwen3.8-27B-UD-Q4_K_XL"
# Qwen3.8 non-thinking recommendation (endpoint README).
DEFAULT_SAMPLING: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
    "max_tokens": 8192,
}
# The handler's timeout on its llama-server call, and our ceiling on submit-and-poll
# (queue time and cold start included).
HANDLER_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 900
POLL_INTERVAL_S = 1.0
MAX_RETRIES = 3

_PENDING = {"IN_QUEUE", "IN_PROGRESS"}
_FAILED = {"FAILED", "CANCELLED", "TIMED_OUT"}
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class QwenClientError(RuntimeError):
    """Anything that stops us getting a usable reply back."""


def model_alias() -> str:
    return os.getenv("QWEN_MODEL_ALIAS", "").strip() or DEFAULT_MODEL_ALIAS


def _base_url() -> str:
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
    if not endpoint_id:
        raise QwenClientError("RUNPOD_ENDPOINT_ID is not set. Add it to backend/.env.")
    return f"https://api.runpod.ai/v2/{endpoint_id}"


def _headers() -> dict[str, str]:
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise QwenClientError("RUNPOD_API_KEY is not set. Add it to backend/.env.")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return isinstance(exc, httpx.TransportError)


def _backoff(attempt: int) -> float:
    return min(2.0**attempt, 30.0) * (0.5 + random.random() / 2)


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            return "RunPod rejected the API key (401). Check RUNPOD_API_KEY."
        if status == 404:
            return f"Endpoint not found (404): {exc.request.url}. Check RUNPOD_ENDPOINT_ID."
        return f"RunPod returned {status}: {exc.response.text[:300]}"
    return f"Could not reach RunPod: {exc!r}"


def build_input(
    *,
    messages: Sequence[dict[str, Any]] | None = None,
    prompt: str | None = None,
    images: Sequence[str] | None = None,
    **sampling: Any,
) -> dict[str, Any]:
    """The handler's `input` object: DEFAULT_SAMPLING overlaid with the caller's non-None knobs."""
    if (prompt is None) == (messages is None):
        raise QwenClientError("Pass exactly one of 'prompt' or 'messages'.")
    payload: dict[str, Any] = {"messages": list(messages)} if messages is not None else {"prompt": prompt}
    if images:
        payload["images"] = list(images)
    payload |= {"internet": False, "think": False, "timeout": HANDLER_TIMEOUT_S}
    payload |= DEFAULT_SAMPLING | {k: v for k, v in sampling.items() if v is not None}
    return payload


def _output(envelope: dict[str, Any]) -> dict[str, Any]:
    status = envelope.get("status")
    if status in _FAILED:
        raise QwenClientError(f"Job {status}: {envelope.get('error') or envelope}")
    output = envelope.get("output") or {}
    if isinstance(output, list):
        output = output[-1] if output else {}
    if not isinstance(output, dict):
        raise QwenClientError(f"Unexpected output shape: {output!r}")
    if output.get("error"):
        raise QwenClientError(f"Handler error: {output['error']}")
    # An 'embed' job answers with `data` (vectors) instead of `text`.
    if output.get("text") is None and output.get("data") is None:
        raise QwenClientError(f"Response contained no text: {envelope}")
    return output


def run(job_input: dict[str, Any]) -> dict[str, Any]:
    """Submit one job and poll until it settles; returns the handler's `output`."""
    url, headers = _base_url(), _headers()
    deadline = time.monotonic() + REQUEST_TIMEOUT_S
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0), headers=headers) as client:

        def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = client.request(method, f"{url}{path}", **kwargs)
                    response.raise_for_status()
                    return response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    if attempt >= MAX_RETRIES or not _retryable(exc):
                        raise QwenClientError(_describe(exc)) from exc
                    time.sleep(_backoff(attempt))
            raise QwenClientError("unreachable")

        envelope = request("POST", "/run", json={"input": job_input})
        job_id = envelope.get("id")
        while envelope.get("status") in _PENDING:
            if time.monotonic() > deadline:
                try:
                    request("POST", f"/cancel/{job_id}")
                except QwenClientError:
                    pass
                raise QwenClientError(f"Job {job_id} still {envelope.get('status')} after {REQUEST_TIMEOUT_S}s; cancelled.")
            time.sleep(POLL_INTERVAL_S)
            envelope = request("GET", f"/status/{job_id}")
    output = _output(envelope)
    _usage.record_runpod(job_input, envelope, output)
    return output


async def arun(job_input: dict[str, Any]) -> dict[str, Any]:
    """Async `run`."""
    url, headers = _base_url(), _headers()
    deadline = time.monotonic() + REQUEST_TIMEOUT_S
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), headers=headers) as client:

        async def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await client.request(method, f"{url}{path}", **kwargs)
                    response.raise_for_status()
                    return response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    if attempt >= MAX_RETRIES or not _retryable(exc):
                        raise QwenClientError(_describe(exc)) from exc
                    await asyncio.sleep(_backoff(attempt))
            raise QwenClientError("unreachable")

        envelope = await request("POST", "/run", json={"input": job_input})
        job_id = envelope.get("id")
        while envelope.get("status") in _PENDING:
            if time.monotonic() > deadline:
                try:
                    await request("POST", f"/cancel/{job_id}")
                except QwenClientError:
                    pass
                raise QwenClientError(f"Job {job_id} still {envelope.get('status')} after {REQUEST_TIMEOUT_S}s; cancelled.")
            await asyncio.sleep(POLL_INTERVAL_S)
            envelope = await request("GET", f"/status/{job_id}")
    output = _output(envelope)
    _usage.record_runpod(job_input, envelope, output)
    return output


def describe_image(prompt: str, image_data_uri: str, **sampling: Any) -> str:
    """One VLM call: `prompt` about one image (a data: URI). Used for figure captioning."""
    return run(build_input(prompt=prompt, images=[image_data_uri], **sampling))["text"].strip()


# ── Embeddings ────────────────────────────────────────────────────────────────

# Qwen3-Embedding is instruction-aware on the query side only; documents go in plain.
DEFAULT_EMBED_INSTRUCTION = "Given a question about electrical machines or generators, retrieve manual passages that answer it"
EMBED_BATCH_SIZE = 32
# The endpoint embeds at most EMBED_CTX_SIZE tokens per text and fails the job on a
# longer one. No character cap is safe — dense tables run to ~1.1 characters per token —
# so a text the endpoint rejects is cut to fit, using the token counts in its error, and
# its vector then covers that first part. EMBED_MAX_CHARS only keeps one job small.
EMBED_MAX_CHARS = 20000
_TOO_LONG = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")
_MIN_EMBED_CHARS = 200


class QwenRunpodEmbeddings(Embeddings):
    """Qwen3-Embedding-4B on the same RunPod endpoint (an `embed` job, v21+).

    Selected with EMBEDDING_PROVIDER="qwen" (rag/vectorstore.py). `dimensions` keeps
    the first N values (Matryoshka) so the vectors fit EMBEDDING_DIMENSION.
    """

    def __init__(self, dimensions: int | None = None, instruction: str | None = None):
        self.dimensions = dimensions
        self.instruction = instruction or os.getenv("QWEN_EMBED_INSTRUCTION", "").strip() or DEFAULT_EMBED_INSTRUCTION

    def _job(self, texts: list[str]) -> dict[str, Any]:
        job: dict[str, Any] = {"embed": [t[:EMBED_MAX_CHARS] or " " for t in texts], "timeout": HANDLER_TIMEOUT_S}
        if self.dimensions:
            job["dimensions"] = self.dimensions
        return job

    @staticmethod
    def _vectors(output: dict[str, Any], expected: int) -> list[list[float]]:
        rows = sorted(output.get("data") or [], key=lambda row: row["index"])
        if len(rows) != expected:
            raise QwenClientError(f"Expected {expected} embeddings, got {len(rows)}.")
        return [row["embedding"] for row in rows]

    def _query_text(self, text: str) -> str:
        return f"Instruct: {self.instruction}\nQuery: {text}"

    @staticmethod
    def _shortened(text: str, error: QwenClientError) -> str | None:
        """`text` cut to fit the context named in a too-long error, or None for any other error."""
        match = _TOO_LONG.search(str(error))
        if not match or len(text) <= _MIN_EMBED_CHARS:
            return None
        tokens, context = int(match.group(1)), int(match.group(2))
        keep = max(_MIN_EMBED_CHARS, min(len(text) // 2, int(len(text) * context / tokens * 0.9)))
        print(f"[embed] {tokens} tokens exceed the embedding context ({context}); "
              f"embedding the first {keep} of {len(text)} characters")
        return text[:keep]

    def _embed_one(self, text: str) -> list[float]:
        text = text[:EMBED_MAX_CHARS]
        while True:
            try:
                return self._vectors(run(self._job([text])), 1)[0]
            except QwenClientError as e:
                text = self._shortened(text, e)
                if text is None:
                    raise

    async def _aembed_one(self, text: str) -> list[float]:
        text = text[:EMBED_MAX_CHARS]
        while True:
            try:
                return self._vectors(await arun(self._job([text])), 1)[0]
            except QwenClientError as e:
                text = self._shortened(text, e)
                if text is None:
                    raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            try:
                vectors += self._vectors(run(self._job(batch)), len(batch))
            except QwenClientError as e:
                if not _TOO_LONG.search(str(e)):
                    raise
                vectors += [self._embed_one(t) for t in batch]  # find and cut the long one(s)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(self._query_text(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            try:
                vectors += self._vectors(await arun(self._job(batch)), len(batch))
            except QwenClientError as e:
                if not _TOO_LONG.search(str(e)):
                    raise
                vectors += [await self._aembed_one(t) for t in batch]
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        return await self._aembed_one(self._query_text(text))


# ── LangChain chat model ──────────────────────────────────────────────────────

_ROLES = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def _to_wire(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [{"role": _ROLES.get(m.type, "user"), "content": m.content} for m in messages]


_JSON_INSTRUCTION = (
    "Respond with only a JSON object that conforms to this JSON schema — no prose, no code fences:\n{schema}"
)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _parse_json(text: str) -> Any:
    text = _FENCE.sub("", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _with_instruction(messages: list[BaseMessage], instruction: str) -> list[BaseMessage]:
    """Append `instruction` to the leading system turn, adding one if needed."""
    if messages and isinstance(messages[0], SystemMessage) and isinstance(messages[0].content, str):
        return [SystemMessage(content=f"{messages[0].content}\n\n{instruction}"), *messages[1:]]
    return [SystemMessage(content=instruction), *messages]


class ChatQwenRunpod(BaseChatModel):
    """LangChain chat model over the RunPod Qwen endpoint (see module docstring)."""

    model_name: str = DEFAULT_MODEL_ALIAS
    temperature: float | None = None
    max_tokens: int | None = None
    # Accepted for drop-in compatibility with ChatOpenAI(streaming=True); the endpoint
    # cannot stream, so the answer arrives as one chunk either way.
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        return "qwen-runpod"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _job(self, messages: list[BaseMessage]) -> dict[str, Any]:
        return build_input(messages=_to_wire(messages), temperature=self.temperature, max_tokens=self.max_tokens)

    @staticmethod
    def _result(output: dict[str, Any]) -> ChatResult:
        completion = output.get("response") or {}
        finish = ((completion.get("choices") or [{}])[0] or {}).get("finish_reason")
        message = AIMessage(content=output["text"], response_metadata={
            "model_name": completion.get("model"), "finish_reason": finish, "usage": completion.get("usage"),
        })
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return self._result(run(self._job(messages)))

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return self._result(await arun(self._job(messages)))

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Runnable:
        """Prompt-level structured output: schema in the system turn, reply validated as `schema`."""
        instruction = _JSON_INSTRUCTION.format(schema=json.dumps(schema.model_json_schema(), ensure_ascii=False))

        def prepare(value: Any) -> list[BaseMessage]:
            return _with_instruction(self._convert_input(value).to_messages(), instruction)

        # The correction goes in a user turn: the handler merges any later system turn into the first.
        def retry_messages(messages: list[BaseMessage], reply: str, error: Exception) -> list[BaseMessage]:
            return [*messages, AIMessage(content=reply),
                    HumanMessage(content=f"That reply was not valid ({error}). {instruction}")]

        def invoke(value: Any) -> BaseModel:
            messages = prepare(value)
            reply = self.invoke(messages).content
            try:
                return schema.model_validate(_parse_json(reply))
            except (ValueError, ValidationError) as error:
                reply = self.invoke(retry_messages(messages, reply, error)).content
                return schema.model_validate(_parse_json(reply))

        async def ainvoke(value: Any) -> BaseModel:
            messages = prepare(value)
            reply = (await self.ainvoke(messages)).content
            try:
                return schema.model_validate(_parse_json(reply))
            except (ValueError, ValidationError) as error:
                reply = (await self.ainvoke(retry_messages(messages, reply, error))).content
                return schema.model_validate(_parse_json(reply))

        return RunnableLambda(invoke, afunc=ainvoke, name=f"{self._llm_type}_structured")
