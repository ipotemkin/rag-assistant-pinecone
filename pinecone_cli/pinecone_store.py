"""Pinecone index management and vector search."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec

from pinecone_cli.config import EMBEDDING_DIMENSION, Settings


@dataclass(frozen=True)
class SearchResult:
    """Single vector search match."""

    vector_id: str
    score: float
    text: str


class PineconeVectorStore:
    """Manage Pinecone indexes and vector operations."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Pinecone(api_key=settings.pinecone_api_key)

    def ensure_index(self, index_name: str) -> None:
        if self._client.has_index(index_name):
            return
        self._client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self._settings.pinecone_cloud,
                region=self._settings.pinecone_region,
            ),
        )

    def upsert_documents(
        self,
        index_name: str,
        texts: list[str],
        embeddings: list[list[float]],
        namespace: str = "",
    ) -> int:
        index = self._client.Index(index_name)
        vectors = [
            (
                str(uuid4()),
                vector,
                {"text": text},
            )
            for text, vector in zip(texts, embeddings)
        ]
        index.upsert(vectors=vectors, namespace=namespace)
        return len(vectors)

    def search(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int = 5,
        namespace: str = "",
    ) -> list[SearchResult]:
        index = self._client.Index(index_name)
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )
        return [
            SearchResult(
                vector_id=match.id,
                score=match.score or 0.0,
                text=str(match.metadata.get("text", "")),
            )
            for match in response.matches
        ]
