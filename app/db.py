from uuid import UUID

import asyncpg

from app.pipeline.summarize import SummaryResult
from app.config import get_settings


async def init_pool() -> asyncpg.Pool:
    global _pool
    settings = get_settings()
    _pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=1,
        max_size=10,
        server_settings={"search_path": settings.db_schema},
    )
    return _pool


async def get_book(pool: asyncpg.Pool, book_id: UUID) -> asyncpg.Record | None:
    return await pool.fetchrow(
        "SELECT id, title, author, source_type, status FROM books WHERE id = $1",
        book_id,
    )


async def set_status(pool: asyncpg.Pool, book_id: UUID, status: str) -> None:
    await pool.execute("UPDATE books SET status = $1 WHERE id = $2", status, book_id)


async def save_summary(
    pool: asyncpg.Pool, book_id: UUID, result: SummaryResult
) -> None:
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
