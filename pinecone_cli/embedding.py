"""OpenAI embeddings via ProxyAPI."""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from pinecone_cli.config import (
    EMBEDDING_MODEL,
    PROXYAPI_BASE_URL,
    Settings,
)


class EmbeddingService(Protocol):
    """Contract for text embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class ProxyApiEmbeddingService:
    """Create embeddings using OpenAI models through ProxyAPI."""

    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(
            api_key=settings.proxyapi_api_key,
            base_url=PROXYAPI_BASE_URL,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
