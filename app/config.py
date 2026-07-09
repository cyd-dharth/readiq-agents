from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env, covering
    core infra, LLM/embedding/search provider selection, pipeline tuning, and the
    chat API."""

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

    upload_dir: str = "/tmp/booksummary_uploads"

    toc_scan_pages: int = 10  # how many leading pages to scan for a table of contents (Tier 1)
    toc_titles_scan_pages: int = 6  # how many leading pages to scan using LLM for chapter titles (Tier 2)
    unreadable_pdf_min_chars: int = 500  # below this many extracted characters, treat the PDF as image-based
    tier4_page_lines_char_limit: int = 4000  # Tier 4 input budget before sampling every other page

    gemini_max_retries: int = 10  # raise once on a paid plan, retries are only needed for free-tier quota waits
    max_concurrent_chapter_summaries: int = 1  # cap parallel chapter LLM calls; raise on a paid plan for full speed
    gemini_rate_limit_per_minute: int = 5  # caps all Gemini calls project-wide to N req/min; set to 0 to disable

    tavily_api_key: str | None = None
    search_provider: str = "tavily"
    min_sources_per_stance: int = 2
    max_sources_per_stance: int = 6

    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 1536

    chat_api_host: str = "0.0.0.0"
    chat_api_port: int = 8001
    max_context_chapters: int = 3
    max_history_messages: int = 8


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance, constructed once on first call."""
    return Settings()
