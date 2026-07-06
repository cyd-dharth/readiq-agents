from __future__ import annotations

import asyncio
from uuid import UUID

import asyncpg
from pydantic import BaseModel

from app import db
from app.llm import LLMClient

_SYSTEM = (
    "You are a careful book analyst. You write original summaries in your own "
    "words and never reproduce the book's text."
)

_CHAPTER_TEXT_SYSTEM = (
    "You are a careful book analyst. You write original chapter summaries in "
    "your own words and never reproduce the book's text."
)

_FROM_CHAPTERS_SYSTEM = (
    "You are a careful book analyst. You write original summaries in your own "
    "words, synthesizing from chapter summaries rather than the book's raw text."
)


class ChapterSummary(BaseModel):
    chapter_number: int
    chapter_title: str | None = None
    summary: str


class SummaryResult(BaseModel):
    one_paragraph_summary: str
    full_summary: str
    chapters: list[ChapterSummary]


class _WholeBook(BaseModel):
    one_paragraph_summary: str
    full_summary: str


class _Chapters(BaseModel):
    chapters: list[ChapterSummary]


class _SequentialChapter(BaseModel):
    summary: str
    updated_digest: str


def _whole_prompt(title: str, author: str | None) -> str:
    by = f" by {author}" if author else ""
    return (
        f"Summarize the book '{title}'{by}. "
        "The one_paragraph_summary is a tight TL;DR of about 60 words. "
        "The full_summary is a thorough, original-words summary of about 400 to 600 words."
    )


def _chapters_prompt(title: str, author: str | None) -> str:
    by = f" by {author}" if author else ""
    return (
        f"List the chapters of '{title}'{by}, in order. For each chapter give a "
        "concise original-words summary of about 80 to 120 words. "
        "Return all chapters in the 'chapters' array."
    )


async def summarize_from_knowledge(
    llm: LLMClient, title: str, author: str | None, max_tokens: int
) -> SummaryResult:
    whole = _WholeBook.model_validate(
        await llm.complete_json(_SYSTEM, _whole_prompt(title, author), max_tokens, _WholeBook)
    )
    # Chapters need more headroom: many books have 15-25 chapters at ~100 words each.
    chapters_resp = _Chapters.model_validate(
        await llm.complete_json(_SYSTEM, _chapters_prompt(title, author), max_tokens * 3, _Chapters)
    )
    return SummaryResult(
        one_paragraph_summary=whole.one_paragraph_summary,
        full_summary=whole.full_summary,
        chapters=chapters_resp.chapters,
    )


def _chapter_summary_prompt(chapter: dict) -> str:
    title_line = f"Chapter title: {chapter['chapter_title']}\n" if chapter.get("chapter_title") else ""
    return (
        f"{title_line}"
        "Summarize this chapter's text in your own original words, in plain "
        "prose of about 80 to 120 words. Do not use JSON, headings, or bullet points.\n\n"
        f"{chapter['raw_text']}"
    )


def _sequential_chapter_prompt(chapter: dict, digest_so_far: str) -> str:
    title_line = f"Chapter title: {chapter['chapter_title']}\n" if chapter.get("chapter_title") else ""
    digest_block = (
        f"Digest of the story so far: {digest_so_far}\n\n"
        if digest_so_far
        else "This is the first chapter, there is no digest yet.\n\n"
    )
    return (
        f"{digest_block}"
        f"{title_line}"
        "Summarize this chapter's text in your own original words, in plain "
        "prose of about 80 to 120 words, for the 'summary' field. Do not use "
        "JSON, headings, or bullet points inside the summary text itself. "
        "Then write an 'updated_digest' of 2 to 3 sentences that folds this "
        "chapter's key developments into the digest so far, so it stays a "
        "short rolling digest of everything up to and including this chapter.\n\n"
        f"{chapter['raw_text']}"
    )


def _from_chapters_whole_prompt(chapter_summaries: list[ChapterSummary]) -> str:
    joined = "\n\n".join(
        f"Chapter {c.chapter_number}"
        + (f" ({c.chapter_title})" if c.chapter_title else "")
        + f": {c.summary}"
        for c in chapter_summaries
    )
    return (
        "Here are original-words summaries of each chapter of a book, in order:\n\n"
        f"{joined}\n\n"
        "Using only these chapter summaries, write a one_paragraph_summary "
        "(a tight TL;DR of about 60 words) and a full_summary (a thorough, "
        "original-words summary of about 400 to 600 words) for the whole book."
    )


async def _summarize_chapter(
    llm: LLMClient,
    chapter: dict,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    pool: asyncpg.Pool,
    book_id: UUID,
) -> ChapterSummary:
    async with semaphore:
        text = await llm.complete(_CHAPTER_TEXT_SYSTEM, _chapter_summary_prompt(chapter), max_tokens)
    await db.save_chapter_summary(pool, book_id, chapter["chapter_number"], text)
    return ChapterSummary(
        chapter_number=chapter["chapter_number"],
        chapter_title=chapter.get("chapter_title"),
        summary=text,
    )


async def _summarize_chapters_sequential(
    llm: LLMClient,
    todo: list[dict],
    max_tokens: int,
    pool: asyncpg.Pool,
    book_id: UUID,
) -> list[ChapterSummary]:
    # Sequential mode does not reconstruct the digest from already-done
    # chapters on resume; it starts empty from wherever the todo list begins.
    digest = ""
    summarized = []
    for chapter in todo:
        result = _SequentialChapter.model_validate(
            await llm.complete_json(
                _CHAPTER_TEXT_SYSTEM,
                _sequential_chapter_prompt(chapter, digest),
                max_tokens,
                _SequentialChapter,
            )
        )
        digest = result.updated_digest
        await db.save_chapter_summary(pool, book_id, chapter["chapter_number"], result.summary)
        summarized.append(
            ChapterSummary(
                chapter_number=chapter["chapter_number"],
                chapter_title=chapter.get("chapter_title"),
                summary=result.summary,
            )
        )
    return summarized


async def summarize_from_chapters(
    llm: LLMClient,
    chapters: list[dict],
    max_tokens: int,
    max_concurrent_chapters: int,
    pool: asyncpg.Pool,
    book_id: UUID,
    sequential: bool = True,
) -> SummaryResult:
    # Chapter stubs (title, number, summary=NULL) already exist in the DB by
    # this point. Skip chapters a prior run already summarized so a crash
    # mid-book does not repeat completed LLM calls.
    existing = await db.get_chapters(pool, book_id)
    done = {row["chapter_number"]: row["summary"] for row in existing if row["summary"] is not None}

    todo = [ch for ch in chapters if ch["chapter_number"] not in done]

    if sequential:
        # Rolling digest continuity depends on strictly ascending chapter order.
        ordered_todo = sorted(todo, key=lambda ch: ch["chapter_number"])
        newly_summarized = await _summarize_chapters_sequential(llm, ordered_todo, max_tokens, pool, book_id)
    else:
        semaphore = asyncio.Semaphore(max_concurrent_chapters)
        newly_summarized = list(
            await asyncio.gather(
                *(_summarize_chapter(llm, ch, max_tokens, semaphore, pool, book_id) for ch in todo)
            )
        )

    chapter_summaries = [
        ChapterSummary(chapter_number=num, chapter_title=None, summary=summary)
        for num, summary in done.items()
    ] + newly_summarized
    chapter_summaries.sort(key=lambda c: c.chapter_number)

    # Restore titles for already-done chapters from the original ingest list.
    titles_by_number = {ch["chapter_number"]: ch.get("chapter_title") for ch in chapters}
    for cs in chapter_summaries:
        if cs.chapter_title is None:
            cs.chapter_title = titles_by_number.get(cs.chapter_number)

    whole = _WholeBook.model_validate(
        await llm.complete_json(
            _FROM_CHAPTERS_SYSTEM,
            _from_chapters_whole_prompt(chapter_summaries),
            max_tokens,
            _WholeBook,
        )
    )
    await db.save_book_summary(pool, book_id, whole.one_paragraph_summary, whole.full_summary)
    return SummaryResult(
        one_paragraph_summary=whole.one_paragraph_summary,
        full_summary=whole.full_summary,
        chapters=chapter_summaries,
    )
