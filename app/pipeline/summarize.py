from __future__ import annotations

from pydantic import BaseModel

from app.llm import LLMClient

_SYSTEM = (
    "You are a careful book analyst. You write original summaries in your own "
    "words and never reproduce the book's text."
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
