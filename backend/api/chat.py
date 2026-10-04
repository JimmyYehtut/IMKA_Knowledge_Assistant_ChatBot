"""GET /api/history — returns the current user's conversation history.

Chat itself is served via the OpenAI-compatible /v1/chat/completions
endpoint (see api/openai_compat.py); this module only exposes the
chat-log read path.
"""
import json
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.auth import get_current_user
from api.database import get_db
from api.db_models import Conversation
from api.schemas import Citation, ConversationOut, MessageOut

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/history", response_model=list[ConversationOut])
async def get_history(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user["user_id"]
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.created_at.desc())
    )
    conversations = result.scalars().all()

    return [
        ConversationOut(
            id=conv.id,
            created_at=conv.created_at.isoformat(),
            persona=conv.persona,
            messages=[
                MessageOut(
                    id=msg.id,
                    role=msg.role,
                    content=msg.content,
                    topic=msg.topic,
                    citations=[Citation(**c) for c in json.loads(msg.citations)],
                    rating=msg.rating,
                    feedback_comment=msg.feedback_comment,
                )
                for msg in conv.messages
            ],
        )
        for conv in conversations
    ]
