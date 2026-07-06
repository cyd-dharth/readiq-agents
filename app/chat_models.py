from __future__ import annotations

from pydantic import BaseModel


class BookContext(BaseModel):
    title: str
    author: str | None
    one_paragraph_summary: str | None
    full_summary: str | None


class ChatRequest(BaseModel):
    book_id: str
    question: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
