"""Centralized, typed application settings.

Every module that needs configuration (API keys, DB connection, ingestion
date range) imports `get_settings()` instead of reading environment
variables directly. This gives us one validated source of truth, makes
missing/misconfigured env vars fail loudly at startup instead of deep
inside a script, and makes tests easy to override via `Settings(**overrides)`.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider -----------------------------------------------------
    # "groq" is the only implemented provider for now. The interface in
    # src/finrag/llm/base.py is written so adding "openai" later is a new
    # class, not a rewrite.
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # --- Ingestion ----------------------------------------------------------
    # Fixed date range so every clone of this repo builds an identical
    # knowledge base. Override via .env only if you deliberately want a
    # different (still fixed) window.
    ingestion_start_date: date = date(2015, 1, 1)
    ingestion_end_date: date = date(2026, 7, 24)

    data_dir: Path = PROJECT_ROOT / "data"

    # --- Database -----------------------------------------------------------
    database_url: str = (
        "postgresql://finrag:finrag@localhost:5432/finrag"
    )

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env vars are read once per process)."""
    return Settings()
