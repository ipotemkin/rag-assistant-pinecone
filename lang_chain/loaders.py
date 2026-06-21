"""Load text documents from strings and files."""

from __future__ import annotations

import json
from pathlib import Path


class DocumentLoader:
    """Load plain-text documents from CLI input sources."""

    def from_text(self, text: str) -> list[str]:
        """Return a single non-empty document."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text must not be empty.")
        return [cleaned]

    def from_file(self, path: Path) -> list[str]:
        """Load documents from a .txt or .json file."""
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._from_txt(path)
        if suffix == ".json":
            return self._from_json(path)

        raise ValueError(
            f"Unsupported file type: {suffix}. Use .txt or .json."
        )

    def _from_txt(self, path: Path) -> list[str]:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"No text lines found in {path}.")
        return lines

    def _from_json(self, path: Path) -> list[str]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = self._extract_json_documents(payload)
        if not documents:
            raise ValueError(f"No documents found in {path}.")
        return documents

    def _extract_json_documents(self, payload: object) -> list[str]:
        if isinstance(payload, list):
            return [self._normalize_item(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("documents", "texts", "items", "data"):
                if key in payload:
                    return self._extract_json_documents(payload[key])
        raise ValueError(
            "JSON must be a list of strings or objects with a text field."
        )

    def _normalize_item(self, item: object) -> str:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                return cleaned
            raise ValueError("JSON string entries must not be empty.")
        if isinstance(item, dict):
            for key in ("text", "content", "body"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        raise ValueError(
            "Each JSON item must be a string or contain text/content."
        )
