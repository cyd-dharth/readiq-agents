from __future__ import annotations

import logging

from app.chat_models import BookContext
from app.llm import LLMClient
from app.pipeline.retrieval import RetrievedContext

log = logging.getLogger(__name__)

_CHAT_SYSTEM_TEMPLATE = """You are a reading assistant for the book \
"{title}"{author_part}.
You answer questions strictly from the provided context below.
The context includes summaries, chapter-level breakdowns, and sourced \
critiques and supporting works about this book.

Rules:
- Only answer from the provided context.
- If the context does not contain enough information, say so clearly \
and suggest which part of the book might address the question.
- Never reproduce large passages. Synthesise and paraphrase.
- Keep answers concise: 2 to 4 sentences for factual questions, \
up to 8 sentences for complex analytical questions.
- If asked about critiques or supporting works, attribute them \
by title and outlet.
- Do not use general knowledge about this book beyond the context."""


def _build_system(ctx: BookContext) -> str:
    """Fill the chat system prompt template with the book's title and author."""
    author_part = f" by {ctx.author}" if ctx.author else ""
    return _CHAT_SYSTEM_TEMPLATE.format(
        title=ctx.title,
        author_part=author_part,
    )


def _build_user_prompt(
    question: str,
    book_ctx: BookContext,
    retrieved: RetrievedContext,
    history: list[dict],
    max_history_messages: int,
) -> str:
    """Build the user prompt from book overview, retrieved chunks, sources, history, and question."""
    parts: list[str] = []

    if book_ctx.one_paragraph_summary:
        parts.append(
            f"Book overview:\n{book_ctx.one_paragraph_summary}"
        )

    if retrieved.chapter_chunks:
        parts.append(
            "Relevant chapter summaries:\n"
            + "\n\n".join(retrieved.chapter_chunks)
        )

    if retrieved.sources_critique:
        parts.append(
            "Critical perspectives (sourced):\n"
            + "\n".join(retrieved.sources_critique)
        )

    if retrieved.sources_support:
        parts.append(
            "Supporting and related works (sourced):\n"
            + "\n".join(retrieved.sources_support)
        )

    context_block = "\n\n===\n\n".join(parts)

    history_block = ""
    if history:
        lines = []
        for msg in history[-max_history_messages:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\n".join(lines)

    prompt_parts = [f"Context:\n{context_block}"]
    if history_block:
        prompt_parts.append(f"Conversation so far:\n{history_block}")
    prompt_parts.append(f"Question: {question}")

    return "\n\n".join(prompt_parts)


async def answer_question(
    llm: LLMClient,
    question: str,
    book_ctx: BookContext,
    retrieved: RetrievedContext,
    history: list[dict],
    max_tokens: int,
    max_history_messages: int = 8,
) -> str:
    """Build the system and user prompts and get a completion from the LLM client."""
    system = _build_system(book_ctx)
    user = _build_user_prompt(question, book_ctx, retrieved, history, max_history_messages)
    return await llm.complete(system, user, max_tokens)
