"""Fetch web pages and split them into text chunks."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
REQUEST_TIMEOUT_SECONDS = 30.0
USER_AGENT = (
    "Mozilla/5.0 (compatible; LangChainCLI/1.0; +https://github.com/)"
)


class UrlDocumentLoader:
    """Download a web page and split its text into chunks."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def from_url(self, url: str) -> list[str]:
        """Fetch URL, extract text, return non-empty chunks."""
        normalized = self._normalize_url(url)
        html = self._fetch_html(normalized)
        text = self._extract_text(html)
        return self._split_text(text, normalized)

    def _normalize_url(self, url: str) -> str:
        cleaned = url.strip()
        if not cleaned:
            raise ValueError("URL must not be empty.")
        parsed = urlparse(cleaned)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must start with http:// or https://.")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        return cleaned

    def _fetch_html(self, url: str) -> str:
        headers = {"User-Agent": USER_AGENT}
        try:
            response = httpx.get(
                url,
                headers=headers,
                follow_redirects=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ValueError(f"Failed to fetch URL: {error}") from error
        return response.text

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        cleaned = _collapse_blank_lines(text)
        if not cleaned:
            raise ValueError("No text content found on the page.")
        return cleaned

    def _split_text(self, text: str, url: str) -> list[str]:
        chunks = self._splitter.split_text(text)
        non_empty = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not non_empty:
            raise ValueError(f"No indexable chunks extracted from {url}.")
        return non_empty


def _collapse_blank_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
