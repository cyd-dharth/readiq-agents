from __future__ import annotations

from uuid import UUID

import asyncpg

from app import db
from app.config import Settings
from app.llm import get_llm
from app.pipeline.summarize import summarize_from_knowledge


async def run_pipeline(pool: asyncpg.Pool, settings: Settings, book_id: str) -> None:
    bid = UUID(book_id)
    book = await db.get_book(pool, bid)
    if book is None:
        raise ValueError(f"Book {book_id} not found")

    await db.set_status(pool, bid, "processing")
    try:
        llm = get_llm(settings)
        if book["source_type"] == "name_only":
            result = await summarize_from_knowledge(llm, book["title"], book["author"], settings.max_output_tokens)
        else:
            # PDF and URL ingest land in the next stage.
            raise NotImplementedError(
                f"source_type '{book['source_type']}' is not supported yet"
            )

        await db.save_summary(pool, bid, result)
        await db.set_status(pool, bid, "ready")
    except Exception:
        await db.set_status(pool, bid, "failed")
        raise
