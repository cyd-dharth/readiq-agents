from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg

from app.pipeline.embed_client import EmbeddingClient

log = logging.getLogger(__name__)

_CRITIQUE_KEYWORDS = frozenset([
    "critic", "criticism", "critique", "against", "oppose",
    "disagree", "wrong", "flaw", "problem", "weakness",
])

_SUPPORT_KEYWORDS = frozenset([
    "similar", "recommend", "related", "support", "agree",
    "reinforce", "extend", "like this", "also read",
])

_SUMMARY_KEYWORDS = frozenset([
    "summary", "overview", "what is this book", "about",
    "main point", "main argument", "thesis",
])


@dataclass
class RetrievedContext:
    chapter_chunks: list[str]
    sources_critique: list[str]
    sources_support: list[str]
    used_vector_search: bool


def _question_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in _CRITIQUE_KEYWORDS):
        return "critique"
    if any(kw in q for kw in _SUPPORT_KEYWORDS):
        return "support"
    if any(kw in q for kw in _SUMMARY_KEYWORDS):
        return "summary"
    return "default"


def _format_source(row) -> str:
    by = f" by {row['author_or_outlet']}" if row["author_or_outlet"] else ""
    return (
        f"{row['title']}{by}: {row['insight']} "
        f"(source: {row['reference_url']})"
    )


async def retrieve(
    pool: asyncpg.Pool,
    embedder: EmbeddingClient,
    book_id: str,
    question: str,
    max_chapters: int = 3,
) -> RetrievedContext:
    qtype = _question_type(question)
    chapter_chunks: list[str] = []
    used_vector = False

    if qtype == "summary":
        # Summary questions skip vector search and rely on the full summary,
        # which the caller already includes via BookContext.
        pass
    else:
        try:
            question_vector = await embedder.embed(question)
            rows = await pool.fetch(
                """
                SELECT chapter_title, summary,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM chapters
                WHERE book_id = $2
                AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                question_vector,
                book_id,
                max_chapters,
            )
            for row in rows:
                label = row["chapter_title"] or "Chapter"
                chunk = f"{label}:\n{row['summary']}"
                chapter_chunks.append(chunk)
            used_vector = True
        except Exception as exc:
            log.warning("Vector search failed: %s", exc)

    # Always fetch sources regardless of question type.
    try:
        critique_rows = await pool.fetch(
            """
            SELECT title, author_or_outlet, insight, reference_url
            FROM sources
            WHERE book_id = $1 AND stance = 'critique'
            ORDER BY created_at
            """,
            book_id,
        )
        support_rows = await pool.fetch(
            """
            SELECT title, author_or_outlet, insight, reference_url
            FROM sources
            WHERE book_id = $1 AND stance = 'support'
            ORDER BY created_at
            """,
            book_id,
        )
    except Exception as exc:
        log.warning("Fetching sources failed: %s", exc)
        critique_rows = []
        support_rows = []

    return RetrievedContext(
        chapter_chunks=chapter_chunks,
        sources_critique=[_format_source(r) for r in critique_rows],
        sources_support=[_format_source(r) for r in support_rows],
        used_vector_search=used_vector,
    )
