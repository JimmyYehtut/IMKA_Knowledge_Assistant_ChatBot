"""Chat model factory shared by intent detection, the RAG pipeline, topics and the Streamlit chain."""
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def llm_provider() -> str:
    """"openai" (default) or "qwen" (Qwen3.8-27B on RunPod, rag/qwen_client.py)."""
    return os.getenv("LLM_PROVIDER", "openai").strip().lower() or "openai"


def chat_model_name() -> str:
    """The name of the model chat_llm() returns, as reported in evaluation traces."""
    if llm_provider() == "qwen":
        from rag.qwen_client import model_alias
        return model_alias()
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def chat_llm(**kwargs) -> BaseChatModel:
    """The chat model selected by LLM_PROVIDER: ChatOpenAI for OPENAI_CHAT_MODEL, or Qwen on RunPod.

    Temperature is sent only when OPENAI_TEMPERATURE is set: some models
    (reasoning-style ones) reject any value but their default, so leaving it
    unset lets the model use its own default instead of failing the request.
    Qwen uses its recommended sampling (0.7) unless OPENAI_TEMPERATURE is set.
    """
    temperature = os.getenv("OPENAI_TEMPERATURE", "").strip()
    if temperature:
        kwargs.setdefault("temperature", float(temperature))
    if llm_provider() == "qwen":
        from rag.qwen_client import ChatQwenRunpod, model_alias
        return ChatQwenRunpod(model_name=model_alias(), **kwargs)
    return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), **kwargs)

