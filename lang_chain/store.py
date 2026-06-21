"""Pinecone vector store via LangChain."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from lang_chain.config import (
    EMBEDDING_DIMENSION,
    TEXT_METADATA_KEY,
    Settings,
)
from lang_chain.embeddings import build_embeddings


@dataclass(frozen=True)
class SearchResult:
    """Single vector search match."""

    vector_id: str
    score: float
    text: str


class LangChainPineconeStore:
    """Manage Pinecone indexes with LangChain vector store."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._embeddings = build_embeddings(settings)

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

    def index_texts(
        self,
        index_name: str,
        texts: list[str],
        namespace: str = "",
    ) -> int:
        self.ensure_index(index_name)
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=self._embeddings,
            index_name=index_name,
            namespace=_normalize_namespace(namespace),
            text_key=TEXT_METADATA_KEY,
            async_req=False,
            pinecone_api_key=self._settings.pinecone_api_key,
        )
        return len(texts)

    def search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        namespace: str = "",
    ) -> list[SearchResult]:
        store = self._build_vector_store(index_name, namespace)
        matches = store.similarity_search_with_score(
            query,
            k=top_k,
            namespace=_normalize_namespace(namespace),
        )
        return [_to_search_result(doc, score) for doc, score in matches]

    def _build_vector_store(
        self,
        index_name: str,
        namespace: str,
    ) -> PineconeVectorStore:
        return PineconeVectorStore(
            index_name=index_name,
            embedding=self._embeddings,
            namespace=_normalize_namespace(namespace),
            text_key=TEXT_METADATA_KEY,
            pinecone_api_key=self._settings.pinecone_api_key,
        )


def _normalize_namespace(namespace: str) -> str | None:
    return namespace or None


def _to_search_result(doc: Document, score: float) -> SearchResult:
    return SearchResult(
        vector_id=doc.id or "unknown",
        score=score,
        text=doc.page_content,
    )
