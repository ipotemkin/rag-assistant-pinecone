"""LangChain embeddings via ProxyAPI."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from lang_chain.config import EMBEDDING_MODEL, PROXYAPI_BASE_URL, Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Create OpenAI embeddings client routed through ProxyAPI."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=settings.proxyapi_api_key,
        base_url=PROXYAPI_BASE_URL,
    )
