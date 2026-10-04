from typing import Literal

from pydantic import BaseModel


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: str
    password: str
    name: str


class UserOut(BaseModel):
    id: str
    email: str
    name: str

    model_config = {"from_attributes": True}


class TokenRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ── Chat ──────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int
    bm25_score: float | None = None
    page: int | None = None
    image_path: str | None = None
    # Absent on messages stored before answers carried inline [n] citations.
    number: int | None = None
    section_path: str | None = None
    chunk_type: str | None = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    topic: str | None = None
    citations: list[Citation]
    rating: Literal["up", "down"] | None = None
    feedback_comment: str | None = None

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    created_at: str
    persona: str | None = None
    messages: list[MessageOut]


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"] | None = None
    comment: str | None = None


class FeedbackResponse(BaseModel):
    message_id: str
    rating: Literal["up", "down"] | None = None
    comment: str | None = None


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentRecord(BaseModel):
    document_id: str
    document_name: str
    version: str
    chunk_count: int
    metadata: dict[str, str]
    status: str
    uploaded_at: str


class IngestResponse(BaseModel):
    total_chunks: int
    documents: list[DocumentRecord]


class HealthResponse(BaseModel):
    status: str
    openai_configured: bool
    qdrant_reachable: bool
