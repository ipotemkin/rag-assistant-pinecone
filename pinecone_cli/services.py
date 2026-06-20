"""Application services orchestrating loaders, embeddings, and Pinecone."""

from __future__ import annotations

from pathlib import Path

from pinecone_cli.embedding import EmbeddingService, ProxyApiEmbeddingService
from pinecone_cli.loaders import DocumentLoader
from pinecone_cli.pinecone_store import PineconeVectorStore, SearchResult
from pinecone_cli.config import Settings


class IndexService:
    """Create a Pinecone index and upsert embedded documents."""

    def __init__(
        self,
        store: PineconeVectorStore,
        embedder: EmbeddingService,
        loader: DocumentLoader,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._loader = loader

    def index_from_text(
        self,
        index_name: str,
        text: str,
        namespace: str = "",
    ) -> int:
        documents = self._loader.from_text(text)
        return self._index_documents(index_name, documents, namespace)

    def index_from_file(
        self,
        index_name: str,
        file_path: Path,
        namespace: str = "",
    ) -> int:
        documents = self._loader.from_file(file_path)
        return self._index_documents(index_name, documents, namespace)

    def _index_documents(
        self,
        index_name: str,
        documents: list[str],
        namespace: str,
    ) -> int:
        self._store.ensure_index(index_name)
        embeddings = self._embedder.embed_texts(documents)
        return self._store.upsert_documents(
            index_name=index_name,
            texts=documents,
            embeddings=embeddings,
            namespace=namespace,
        )


class SearchService:
    """Search for relevant documents in a Pinecone index."""

    def __init__(
        self,
        store: PineconeVectorStore,
        embedder: EmbeddingService,
    ) -> None:
        self._store = store
        self._embedder = embedder

    def search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        namespace: str = "",
    ) -> list[SearchResult]:
        query_vector = self._embedder.embed_query(query)
        return self._store.search(
            index_name=index_name,
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
        )


def build_index_service(settings: Settings) -> IndexService:
    """Factory for index operations."""
    return IndexService(
        store=PineconeVectorStore(settings),
        embedder=ProxyApiEmbeddingService(settings),
        loader=DocumentLoader(),
    )


def build_search_service(settings: Settings) -> SearchService:
    """Factory for search operations."""
    return SearchService(
        store=PineconeVectorStore(settings),
        embedder=ProxyApiEmbeddingService(settings),
    )
