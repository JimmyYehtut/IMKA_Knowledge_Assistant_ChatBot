"""Chat model factory shared by intent detection, the RAG pipeline, and the Streamlit chain."""
import os

from langchain_openai import ChatOpenAI


def chat_llm(**kwargs) -> ChatOpenAI:
    """ChatOpenAI for OPENAI_CHAT_MODEL.

    Temperature is sent only when OPENAI_TEMPERATURE is set: some models
    (reasoning-style ones) reject any value but their default, so leaving it
    unset lets the model use its own default instead of failing the request.
    """
    temperature = os.getenv("OPENAI_TEMPERATURE", "").strip()
    if temperature:
        kwargs.setdefault("temperature", float(temperature))
    return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), **kwargs)
