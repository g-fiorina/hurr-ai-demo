from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://hurr:hurr_secret@localhost:5432/hurr_ai")
    voyage_api_key: str = os.getenv("VOYAGE_API_KEY", "")
    jina_api_key: str = os.getenv("JINA_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")


settings = Settings()

PROVIDER_DIMENSIONS: dict[str, int] = {
    # "voyage_1024": 1024,  # disabled – max search quality
    "voyage_2048": 2048,
    # "jina_1024": 1024,  # disabled – max search quality
    "jina_2048": 2048,
    # "cohere_1024": 1024,  # disabled – max search quality
    "cohere_1536": 1536,
}
