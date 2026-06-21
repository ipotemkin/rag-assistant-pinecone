"""DuckDuckGo internet search tool."""

from __future__ import annotations

from langchain_core.tools import tool

from ddgs import DDGS

MAX_WEB_RESULTS = 5


@tool
def search_internet(query: str) -> str:
    """Search the internet via DuckDuckGo when the knowledge base has no answer."""
    try:
        hits = DDGS().text(query, max_results=MAX_WEB_RESULTS)
    except Exception as error:  # noqa: BLE001 - surface error to the LLM
        return f"Internet search failed: {error}"

    if not hits:
        return "No internet results found."

    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        title = hit.get("title", "")
        body = hit.get("body", "")
        href = hit.get("href", "")
        lines.append(f"{index}. {title}\n   {body}\n   {href}")
    return "\n".join(lines)
