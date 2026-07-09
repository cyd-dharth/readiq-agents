from __future__ import annotations

from pydantic import BaseModel


class BookContext(BaseModel):
    """Book identity and summaries fetched server side from the DB, used to ground
    chat answers; never populated from caller-supplied data."""

    title: str
    author: str | None
    one_paragraph_summary: str | None
    full_summary: str | None


class ChatRequest(BaseModel):
    """Request body for POST /chat. Carries only the book id, question, and optional
    prior turns; the book's own content is always looked up from the DB, never trusted
    from the caller."""

    book_id: str
    question: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    """Response body for POST /chat containing the generated answer."""

    answer: str
