from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = ""
    sentry_dsn: str = ""
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "gemma3:4b"
    ollama_embed_model: str = "nomic-embed-text"
    keepsake_person_name: str = "Dadi"
    keepsake_person_relation: str = "grandmother"
    # nomic-embed-text outputs 768-d vectors
    embed_dims: int = 768
    retrieve_k: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
