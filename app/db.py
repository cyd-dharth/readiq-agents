from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import asyncpg

from app.config import get_settings

if TYPE_CHECKING:
    from app.pipeline.summarize import SummaryResult


async def init_pool() -> asyncpg.Pool:
    """Create the worker's asyncpg pool, registering pgvector so vector columns can be bound."""
    global _pool
    from pgvector.asyncpg import register_vector

    settings = get_settings()
    _pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=1,
        max_size=10,
        server_settings={"search_path": settings.db_schema},
        init=register_vector,
    )
    return _pool


async def get_book(pool: asyncpg.Pool, book_id: UUID) -> asyncpg.Record | None:
    """Fetch the core book row (identity, source, and status fields) by id."""
    return await pool.fetchrow(
        "SELECT id, title, author, source_type, source_ref, status, research_status FROM books WHERE id = $1",
        book_id,
    )


async def set_status(pool: asyncpg.Pool, book_id: UUID, status: str) -> None:
    """Update books.status, e.g. to processing, ready, or failed."""
    await pool.execute("UPDATE books SET status = $1 WHERE id = $2", status, book_id)


async def create_chapter_stubs(pool: asyncpg.Pool, book_id: UUID, chapters: list[dict]) -> None:
    """Insert a row per detected chapter with summary left NULL.

    Uses ON CONFLICT DO NOTHING on (book_id, chapter_number) so a resumed
    run does not wipe out summaries already written by a prior attempt.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            for ch in chapters:
                await conn.execute(
                    """
                    INSERT INTO chapters (book_id, chapter_number, chapter_title)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (book_id, chapter_number) DO NOTHING
                    """,
                    book_id,
                    ch["chapter_number"],
                    ch.get("chapter_title"),
                )


async def get_chapters(pool: asyncpg.Pool, book_id: UUID) -> list[asyncpg.Record]:
    """Fetch all chapters for a book, including any with a NULL summary."""
    return await pool.fetch(
        "SELECT chapter_number, chapter_title, summary FROM chapters WHERE book_id = $1",
        book_id,
    )


async def save_chapter_summary(
    pool: asyncpg.Pool, book_id: UUID, chapter_number: int, summary: str
) -> None:
    """Write a single chapter's summary immediately, so progress persists as each chapter finishes."""
    await pool.execute(
        "UPDATE chapters SET summary = $1 WHERE book_id = $2 AND chapter_number = $3",
        summary,
        book_id,
        chapter_number,
    )


async def save_book_summary(
    pool: asyncpg.Pool, book_id: UUID, one_paragraph_summary: str, full_summary: str
) -> None:
    """Write the whole-book one-paragraph and full summaries once synthesized from chapter summaries."""
    await pool.execute(
        """
        UPDATE books
        SET one_paragraph_summary = $1, full_summary = $2
        WHERE id = $3
        """,
        one_paragraph_summary,
        full_summary,
        book_id,
    )


async def get_book_summary(pool: asyncpg.Pool, book_id: UUID) -> dict | None:
    """Fetch a book's title, author, and summaries for the chat API to build BookContext from."""
    row = await pool.fetchrow(
        """
        SELECT id, title, author, one_paragraph_summary, full_summary
        FROM books
        WHERE id = $1
        """,
        book_id,
    )
    return dict(row) if row is not None else None


async def save_sources(pool: asyncpg.Pool, book_id: UUID, sources: list[dict]) -> None:
    """Replace all sources for this book so re-running research is idempotent."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM sources WHERE book_id = $1", book_id)
            if not sources:
                return
            await conn.executemany(
                """
                INSERT INTO sources (
                    book_id, stance, source_type, title, author_or_outlet,
                    reference_url, insight, about_living_person, verified
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                [
                    (
                        book_id,
                        s["stance"],
                        s["source_type"],
                        s["title"],
                        s["author_or_outlet"],
                        s["reference_url"],
                        s["insight"],
                        s["about_living_person"],
                        s["verified"],
                    )
                    for s in sources
                ],
            )


async def set_research_status(pool: asyncpg.Pool, book_id: UUID, status: str) -> None:
    """Update books.research_status, e.g. to pending, completed, failed, or skipped."""
    await pool.execute("UPDATE books SET research_status = $1 WHERE id = $2", status, book_id)


async def save_summary(
    pool: asyncpg.Pool, book_id: UUID, result: SummaryResult
) -> None:
    """Used by the name_only path, which has no per-chapter resume needs."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                UPDATE books
                SET one_paragraph_summary = $1, full_summary = $2
                WHERE id = $3
                """,
                result.one_paragraph_summary,
                result.full_summary,
                book_id,
            )
            # Replace chapters so re-running a book is idempotent.
            await conn.execute("DELETE FROM chapters WHERE book_id = $1", book_id)
            for ch in result.chapters:
                await conn.execute(
                    """
                    INSERT INTO chapters (book_id, chapter_number, chapter_title, summary)
                    VALUES ($1, $2, $3, $4)
                    """,
                    book_id,
                    ch.chapter_number,
                    ch.chapter_title,
                    ch.summary,
                )
