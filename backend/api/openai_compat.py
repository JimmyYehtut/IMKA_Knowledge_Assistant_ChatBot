"""OpenAI-compatible chat completions API — GET /v1/models, POST /v1/chat/completions.

Wire format matches OpenAI's Chat Completions API (streaming via SSE and
non-streaming), so any OpenAI-compatible client can talk to this backend.
Extensions beyond the spec (ignored by strict OpenAI clients, used by our
own frontend): `conversation_id` and `message_id` for grouping/targeting
chat-log rows, `persona` (answer persona, fixed when the conversation is
created), `topic` (short label of the user's question), and `citations` with
the RAG source documents.

Evaluation extensions (opt-in, unused by the frontend): `document_ids` restricts
retrieval to those documents, and `include_context: true` (non-streaming only)
adds a `rag` object to the response — intent, rewritten query, chat model, the
stage 2 candidates, the reranked chunks and the context passages the answer was
generated from. See test/README.md.
"""
import json
import time
import uuid
from typing import Annotated, AsyncIterator, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from api.auth import get_current_user
from api.database import get_db
from api.db_models import Conversation, Message
from rag.intent import classify_intent
from rag import usage
from rag.llm import chat_model_name
from rag.vectorstore import embedding_model_name
from rag.pipeline import PERSONAS, build_answer_chain, persona_instructions, prepare_context, select_cited

router = APIRouter(tags=["openai-compat"])

MODEL_ID = "rag-pipeline"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    conversation_id: str | None = None
    persona: str | None = None
    document_ids: list[str] | None = None
    include_context: bool = False


def _extract_query(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(status_code=400, detail="No user message found")


async def _get_or_create_conversation(
    db: AsyncSession, user_id: str, conversation_id: str | None, persona: str | None = None
) -> Conversation:
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    conversation = Conversation(user_id=user_id, persona=persona if persona in PERSONAS else None)
    db.add(conversation)
    await db.flush()
    return conversation


async def _generate_answer(
    query: str,
    persona: str | None = None,
    trace_metadata: dict | None = None,
    document_ids: list[str] | None = None,
    trace: dict | None = None,
) -> tuple[str, list[dict], str]:
    """Runs intent detection first, then the RAG pipeline if needed. Returns (answer, citations, topic).

    When `trace` is a dict it is filled with the evaluation trace (the `rag` response object).
    """
    intent = await classify_intent(query, trace_metadata)
    if trace is not None:
        trace.update(
            intent=intent.intent,
            model=chat_model_name(),
            embedding_model=embedding_model_name(),
            document_ids=document_ids or [],
            rewritten_query=None,
            candidates=[],
            chunks=[],
            contexts=[],
        )
    if intent.intent != "knowledge" and intent.reply:
        return intent.reply, [], ""

    context, citations, rewritten = await prepare_context(query, trace_metadata, document_ids, trace)
    if trace is not None:
        trace["rewritten_query"] = rewritten
    if not context:
        return "No relevant documents found for your query in the knowledge base.", [], intent.topic

    chain = build_answer_chain()
    answer = await chain.ainvoke(
        {"context": context, "question": query, "persona_instructions": persona_instructions(persona)},
        config={"run_name": "rag_answer_generation", "tags": ["rag", "stage5"], "metadata": trace_metadata or {}},
    )
    return answer, select_cited(answer, citations), intent.topic


async def _persist_exchange(
    db: AsyncSession, conversation: Conversation, query: str, answer: str, citations: list[dict], topic: str = ""
) -> str:
    """Persists the user+assistant messages. Returns the assistant message's id."""
    assistant_id = str(uuid.uuid4())
    db.add(Message(conversation_id=conversation.id, role="user", content=query, topic=topic[:120]))
    db.add(
        Message(
            id=assistant_id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            citations=json.dumps(citations),
        )
    )
    await db.commit()
    return assistant_id


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "local"}],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user["user_id"]
    query = _extract_query(body.messages)
    conversation = await _get_or_create_conversation(db, user_id, body.conversation_id, body.persona)
    trace_metadata = {"conversation_id": conversation.id, "user_id": user_id, "persona": conversation.persona}

    if body.stream:
        return StreamingResponse(
            _stream_sse(query, conversation, db, trace_metadata, body.document_ids),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    trace = {} if body.include_context else None
    calls = usage.collect() if trace is not None else None
    answer, citations, topic = await _generate_answer(
        query, conversation.persona, trace_metadata, body.document_ids, trace
    )
    if trace is not None:
        trace["usage"] = usage.summary(calls)
    message_id = await _persist_exchange(db, conversation, query, answer, citations, topic)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "conversation_id": conversation.id,
        "message_id": message_id,
        "topic": topic,
        "citations": citations,
    }
    if trace is not None:
        response["rag"] = trace
    return response


def _chunk(completion_id: str, delta: dict, finish_reason: str | None = None, extra: dict | None = None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if extra:
        payload.update(extra)
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_sse(
    query: str,
    conversation: Conversation,
    db: AsyncSession,
    trace_metadata: dict | None = None,
    document_ids: list[str] | None = None,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    yield _chunk(completion_id, {"role": "assistant"})

    full_answer = ""
    citations: list[dict] = []

    intent = await classify_intent(query, trace_metadata)
    topic = ""
    if intent.intent != "knowledge" and intent.reply:
        full_answer = intent.reply
        yield _chunk(completion_id, {"content": full_answer})
    else:
        topic = intent.topic
        context, citations, rewritten = await prepare_context(query, trace_metadata, document_ids)
        if not context:
            full_answer = "No relevant documents found for your query in the knowledge base."
            yield _chunk(completion_id, {"content": full_answer})
        else:
            chain = build_answer_chain()
            stream_config = {
                "run_name": "rag_answer_generation",
                "tags": ["rag", "stage5"],
                "metadata": trace_metadata or {},
            }
            chain_input = {
                "context": context,
                "question": query,
                "persona_instructions": persona_instructions(conversation.persona),
            }
            async for token in chain.astream(chain_input, config=stream_config):
                if token:
                    full_answer += token
                    yield _chunk(completion_id, {"content": token})
            citations = select_cited(full_answer, citations)

    message_id = await _persist_exchange(db, conversation, query, full_answer, citations, topic)

    yield _chunk(
        completion_id,
        {},
        finish_reason="stop",
        extra={
            "conversation_id": conversation.id,
            "message_id": message_id,
            "topic": topic,
            "citations": citations,
        },
    )
    yield "data: [DONE]\n\n"
