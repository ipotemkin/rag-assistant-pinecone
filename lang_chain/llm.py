"""LangChain chat model via ProxyAPI."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from lang_chain.config import PROXYAPI_BASE_URL, Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create OpenAI chat client routed through ProxyAPI."""
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.proxyapi_api_key,
        base_url=PROXYAPI_BASE_URL,
        temperature=0.2,
    )
