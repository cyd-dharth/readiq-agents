from __future__ import annotations

import logging
from uuid import UUID

import asyncpg

from app.pipeline.embed_client import EmbeddingClient

log = logging.getLogger(__name__)


async def embed_chapters(
    pool: asyncpg.Pool,
    book_id: UUID,
    embedder: EmbeddingClient,
    model_name: str,
) -> None:
    """Embed all summarized chapters for a book and write vectors back to Postgres.

    Fetches chapters with a non null, non empty summary, embeds them in one
    batch call, then writes embedding and model per chapter inside a single
    transaction. Idempotent: reruns overwrite existing embeddings. Never
    raises, logs a warning and returns if there is nothing to embed or if the
    embedding API call fails.
    """
    rows = await pool.fetch(
        """
        SELECT id, summary FROM chapters
        WHERE book_id = $1
        AND summary IS NOT NULL
        AND summary != ''
        ORDER BY chapter_number
        """,
        book_id,
    )

    if not rows:
        log.warning("No chapter summaries found to embed for book %s", book_id)
        return

    chapter_ids = [r["id"] for r in rows]
    texts = [r["summary"] for r in rows]

    log.info("Embedding %d chapters for book %s", len(texts), book_id)

    try:
        vectors = await embedder.embed_many(texts)
    except Exception as exc:
        log.warning(
            "Embedding API call failed for book %s: %s",
            book_id,
            exc,
        )
        return

    async with pool.acquire() as conn:
        async with conn.transaction():
            for chapter_id, vector in zip(chapter_ids, vectors):
                await conn.execute(
                    """
                    UPDATE chapters
                    SET embedding = $1, model = $2
                    WHERE id = $3
                    """,
                    vector,
                    model_name,
                    chapter_id,
                )

    log.info(
        "Embed complete: %d chapters written for book %s",
        len(vectors),
        book_id,
    )
