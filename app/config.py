from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:root@localhost:5432/postgres"
    db_schema: str = "booksummary"
    redis_url: str = "redis://localhost:6379/0"
    job_queue: str = "book_generation"

    llm_provider: str = "anthropic"  # anthropic | openai | gemini
    llm_model: str = "claude-sonnet-4-6"
    max_output_tokens: int = 4000   # Chapter summaries take 3x the space of the one-paragraph summary and full summary combined.

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
