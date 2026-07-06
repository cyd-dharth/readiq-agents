from __future__ import annotations

import logging
from uuid import UUID

import asyncpg

from app import db
from app.config import Settings
from app.llm import get_llm
from app.pipeline.embed import embed_chapters
from app.pipeline.embed_client import get_embedding_client
from app.pipeline.ingest import ingest_pdf
from app.pipeline.ingest_epub import ingest_epub
from app.pipeline.research import run_research
from app.pipeline.search import get_search_client
from app.pipeline.summarize import summarize_from_chapters, summarize_from_knowledge

log = logging.getLogger(__name__)


async def run_pipeline(pool: asyncpg.Pool, settings: Settings, book_id: str) -> None:
    bid = UUID(book_id)  # book id
    book = await db.get_book(pool, bid)
    if book is None:
        raise ValueError(f"Book {book_id} not found")

    if book["status"] in ["ready", "published"]:
        if book["research_status"] == "completed":
            log.info("Book %s already summarized and researched, nothing to do", book_id)
            return
        
        log.info(
            "Book %s already summarized (status=ready), research_status=%s, running research only",
            book_id, book["research_status"],
        )
        llm = get_llm(settings)
        await _run_embed_stage(pool, settings, bid, book_id)
        await _run_research_stage(pool, settings, llm, bid, book_id, book)
        return

    await db.set_status(pool, bid, "processing")
    try:
        llm = get_llm(settings)
        if book["source_type"] == "name_only":
            result = await summarize_from_knowledge(llm, book["title"], book["author"], settings.max_output_tokens)
            await db.save_summary(pool, bid, result)
        elif book["source_type"] == "pdf":
            if not book["source_ref"]:
                raise ValueError("source_ref is null for a pdf book")
            chapters = await ingest_pdf(book["source_ref"], llm, settings.max_output_tokens)
            await db.create_chapter_stubs(pool, bid, chapters)
            # Chapter summaries and the whole-book summary are written to the
            # DB incrementally inside summarize_from_chapters, so a crash
            # mid-book resumes from whichever chapters are still summary=NULL
            # instead of repeating the whole pipeline.
            await summarize_from_chapters(
                llm, chapters, settings.max_output_tokens, settings.max_concurrent_chapter_summaries, pool, bid)
        elif book["source_type"] == "epub":
            if not book["source_ref"]:
                raise ValueError("source_ref is null for an epub book")
            chapters = ingest_epub(book["source_ref"])
            await db.create_chapter_stubs(pool, bid, chapters)
            await summarize_from_chapters(
                llm, chapters, settings.max_output_tokens, settings.max_concurrent_chapter_summaries, pool, bid)
        else:
            # URL ingest lands in a later stage.
            raise NotImplementedError(
                f"source_type '{book['source_type']}' is not supported yet"
            )

        await db.set_status(pool, bid, "ready")
    except Exception:
        await db.set_status(pool, bid, "failed")
        raise

    # Embed and research both run after the book is already marked ready, so
    # a failure in either never leaves the book stuck in processing or flips
    # it back to failed. The book stays ready regardless of their outcome.
    await _run_embed_stage(pool, settings, bid, book_id)
    await _run_research_stage(pool, settings, llm, bid, book_id, book)


async def _run_embed_stage(pool, settings, bid, book_id) -> None:
    try:
        embedder = get_embedding_client(settings)
        await embed_chapters(
            pool,
            bid,
            embedder,
            settings.embedding_model,
        )
    except Exception:
        log.exception(
            "Embed stage failed for book %s. Continuing.",
            book_id,
        )


async def _run_research_stage(pool, settings, llm, bid, book_id, book) -> None:
    try:
        search = get_search_client(settings)
        book_summary = await db.get_book_summary(pool, bid)
        research_result = await run_research(
            llm=llm,
            search=search,
            title=book["title"],
            author=book["author"],
            summary=(book_summary or {}).get("one_paragraph_summary") or "",
            max_tokens=settings.max_output_tokens,
            min_items=settings.min_sources_per_stance,
            max_items=settings.max_sources_per_stance,
        )
        all_sources = [
            {
                "stance": item.stance,
                "source_type": item.source_type,
                "title": item.title,
                "author_or_outlet": item.author_or_outlet,
                "reference_url": item.reference_url,
                "insight": item.insight,
                "about_living_person": item.about_living_person,
                "verified": item.verified,
            }
            for item in (research_result.critiques + research_result.supports)
        ]
        await db.save_sources(pool, bid, all_sources)
        await db.set_research_status(pool, bid, "completed")
        log.info(
            "Research complete: %d critiques, %d supports for book %s",
            len(research_result.critiques),
            len(research_result.supports),
            book_id,
        )
    except Exception:
        log.exception(
            "Research stage failed for book %s. Book remains ready without sources.",
            book_id,
        )
        await db.set_research_status(pool, bid, "failed")
