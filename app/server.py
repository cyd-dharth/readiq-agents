import logging
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
from fastapi import FastAPI, HTTPException

from app import db
from app.chat_models import BookContext, ChatRequest, ChatResponse
from app.config import get_settings
from app.llm import get_llm
from app.pipeline.chat import answer_question
from app.pipeline.embed_client import get_embedding_client
from app.pipeline.retrieval import retrieve

log = logging.getLogger(__name__)
_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    from pgvector.asyncpg import register_vector

    settings = get_settings()
    _pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=1,
        max_size=5,
        server_settings={"search_path": settings.db_schema},
        init=lambda conn: register_vector(conn),
    )
    yield
    if _pool:
        await _pool.close()


app = FastAPI(title="Book Summary Agents Internal API", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB pool not ready")

    try:
        book_id = UUID(req.book_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid book_id")

    book_summary = await db.get_book_summary(_pool, book_id)
    if book_summary is None:
        raise HTTPException(status_code=404, detail="Book not found")

    book_ctx = BookContext(
        title=book_summary["title"],
        author=book_summary["author"],
        one_paragraph_summary=book_summary["one_paragraph_summary"],
        full_summary=book_summary["full_summary"],
    )

    try:
        embedder = get_embedding_client(settings)
        llm = get_llm(settings)

        retrieved = await retrieve(
            pool=_pool,
            embedder=embedder,
            book_id=req.book_id,
            question=req.question,
            max_chapters=settings.max_context_chapters,
        )

        answer = await answer_question(
            llm=llm,
            question=req.question,
            book_ctx=book_ctx,
            retrieved=retrieved,
            history=req.history,
            max_tokens=settings.max_output_tokens,
            max_history_messages=settings.max_history_messages,
        )

        return ChatResponse(answer=answer)

    except Exception as exc:
        log.exception("Chat request failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat failed")


@app.get("/health")
async def health():
    return {"status": "ok"}
