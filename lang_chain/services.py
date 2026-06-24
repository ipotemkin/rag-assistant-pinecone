"""Application services orchestrating loaders and LangChain store."""

from __future__ import annotations

from pathlib import Path

from lang_chain.config import Settings
from lang_chain.loaders import DocumentLoader
from lang_chain.store import LangChainPineconeStore, SearchResult
from lang_chain.url_loader import UrlDocumentLoader


class IndexService:
    """Create a Pinecone index and upsert embedded documents."""

    def __init__(
        self,
        store: LangChainPineconeStore,
        loader: DocumentLoader,
        url_loader: UrlDocumentLoader | None = None,
    ) -> None:
        self._store = store
        self._loader = loader
        self._url_loader = url_loader or UrlDocumentLoader()

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

    def index_from_url(
        self,
        index_name: str,
        url: str,
        namespace: str = "",
    ) -> int:
        documents = self._url_loader.from_url(url)
        return self._index_documents(index_name, documents, namespace)

    def _index_documents(
        self,
        index_name: str,
        documents: list[str],
        namespace: str,
    ) -> int:
        return self._store.index_texts(
            index_name=index_name,
            texts=documents,
            namespace=namespace,
        )


class SearchService:
    """Search for relevant documents in a Pinecone index."""

    def __init__(self, store: LangChainPineconeStore) -> None:
        self._store = store

    def search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        namespace: str = "",
    ) -> list[SearchResult]:
        return self._store.search(
            index_name=index_name,
            query=query,
            top_k=top_k,
            namespace=namespace,
        )


def build_index_service(
    settings: Settings,
    url_loader: UrlDocumentLoader | None = None,
) -> IndexService:
    """Factory for index operations."""
    return IndexService(
        store=LangChainPineconeStore(settings),
        loader=DocumentLoader(),
        url_loader=url_loader,
    )


def build_search_service(settings: Settings) -> SearchService:
    """Factory for search operations."""
    return SearchService(store=LangChainPineconeStore(settings))
