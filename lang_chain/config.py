"""Application configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openai/v1"
TEXT_METADATA_KEY = "text"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from .env and environment."""

    pinecone_api_key: str
    proxyapi_api_key: str
    pinecone_cloud: str
    pinecone_region: str


def load_settings() -> Settings:
    """Load and validate required settings."""
    load_dotenv()
    return Settings(
        pinecone_api_key=_require("PINECONE_API_KEY"),
        proxyapi_api_key=_require("PROXYAPI_API_KEY"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
    )


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value
