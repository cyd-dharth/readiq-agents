# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is the full context for this repo. Read it completely before writing
any code, suggesting changes, or answering questions. Every decision here was
made deliberately. Do not suggest alternatives unless asked.

---

## What this service is

The AI generation pipeline for a book summary platform. It runs as a worker
process, consuming jobs from a Redis queue, running the pipeline for each book,
and writing results back to the shared Postgres database.

This is one of three repos:

- `booksummary-backend`: FastAPI controller, public API, job dispatch, DB schema
- `booksummary-agents` (this repo): the AI generation pipeline
- `booksummary-frontend`: Astro website that renders published content

Infrastructure (Postgres and Redis) is brought up by the backend repo's
`docker-compose.yml`. This service connects to both via env vars.

---

## Project context

### What the platform does

Users visit a website showing AI-generated book summaries. Each book page has a
one-paragraph TL;DR, a full summary, chapter-by-chapter summaries, and later
(V1.1) sourced critique and support, and (V1.2) a RAG-powered chat. The pipeline
runs locally; the live site only reads published rows.

### Phasing

- V1 (current): whole-book summary, chapter summaries, status written to Postgres.
- V1.1 (next): sourced critique and support with reference links, stances, and a
  verified flag. A new `sources` table in Postgres (defined in the backend repo).
- V1.2 (later): chapter-summary embeddings written to pgvector for RAG chat.
- Phase 2 (future): audio generation and YouTube video pipeline.

---

## Running locally

Prerequisites: Docker running (started from the backend repo), Python 3.13+, `uv`.

```
# 1. Infrastructure must be running (from booksummary-backend)
#    docker compose up -d

# 2. Install dependencies
uv sync

# 3. Configure
cp .env.example .env
# Set LLM_PROVIDER and the matching API key
# DATABASE_URL and REDIS_URL point to the backend's docker compose services

# 4. Run the worker
uv run python main.py
```

There are no tests yet. There is no linter configured. The worker logs every job
it picks up, its outcome, and any exceptions with full tracebacks.

A `Dockerfile` is present for containerized deployment; it is not used in the
local development flow.

---

## Architecture

```
Redis queue  <--  backend enqueues {book_id}
     |
     v
main.py  (BRPOP loop, one job at a time)
     |
     v
pipeline/runner.py  (orchestrates stages, owns status transitions)
     |
     |-- pipeline/summarize.py   (LLM calls: whole-book + per-chapter)
     |-- pipeline/ingest.py      (PDF parse + chapter detection) [next]
     |-- pipeline/embed.py       (pgvector embeddings) [V1.2]
     |-- pipeline/research.py    (critique + support sourcing) [V1.1]
     |
     v
Postgres  (writes summaries and chapters, flips status to ready or failed)
```

The worker is a single async process. It blocks on Redis with `BRPOP`, picks up
one job, runs the full pipeline for that book, then blocks again. One book at a
time is correct for V1; concurrency can be added later if needed.

---

## Database schema (read-only for this service)

This service does not own the schema. The DDL lives in the backend repo at
`db/schema_website_v1.sql`. This service only reads and writes rows.

All queries execute within the `booksummary` Postgres schema, set via
`search_path` in the connection pool (controlled by `db_schema` in `config.py`).

### books (relevant columns)

- `id` uuid PK
- `title`, `author`
- `source_type`: `name_only` | `pdf` | `url`
- `source_ref`: path to uploaded file, NULL for name_only
- `one_paragraph_summary`: written by this service after generation
- `full_summary`: written by this service after generation
- `status`: this service reads `processing` and writes `ready` or `failed`
- `copyright_status`: `public_domain` | `in_copyright` | `permission_granted`

### chapters (written by this service)

- `id` uuid PK, `book_id` FK (cascade delete)
- `chapter_number`, `chapter_title`, `summary`, `word_count`
- `embedding` vector(1536): written in V1.2, NULL for now
- `model`: embedding model id, for re-embedding on model change
- UNIQUE on (book_id, chapter_number)

The pipeline deletes existing chapters before inserting new ones so re-running
a book is idempotent.

---

## Status transitions owned by this service

```
processing  ->  ready    (pipeline completed successfully)
processing  ->  failed   (any unhandled exception in the pipeline)
```

The backend owns all other transitions. This service never sets `published`.

On failure, the exception is logged with full traceback and the book status is
set to `failed`. The worker does not crash; it logs and moves on to the next job.

---

## LLM client

A provider-agnostic interface in `app/llm.py`. One abstract base class
(`LLMClient`) with a single method `complete(system, user, max_tokens) -> str`.
Three implementations behind a factory function `get_llm(settings)`:

- `AnthropicClient`: uses `anthropic.AsyncAnthropic`, `messages.create`
- `OpenAIClient`: uses `openai.AsyncOpenAI`, `chat.completions.create`
- `GeminiClient`: uses `google.genai.Client`, `client.aio.models.generate_content`
  with `types.GenerateContentConfig(system_instruction=..., max_output_tokens=...)`
  Note: uses the current `google-genai` SDK, not the deprecated `google-generativeai`.

The provider is selected by `LLM_PROVIDER` in env. SDK imports are lazy (inside
`__init__`) so only the installed provider's package is needed. All three guard
against a missing API key with a clear `RuntimeError` at client construction time.

To add a new provider: create a new class inheriting `LLMClient`, implement
`complete`, add a branch in `get_llm`, add the key to `Settings`, and add it
to the env example. Do not change the `LLMClient` interface.

---

## File structure

```
main.py              entry point: asyncio.run, Redis BRPOP loop, job dispatch
app/
  config.py          pydantic-settings: DATABASE_URL, REDIS_URL, JOB_QUEUE,
                     LLM_PROVIDER, LLM_MODEL, MAX_OUTPUT_TOKENS,
                     ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
                     DB_SCHEMA (default: "booksummary")
  llm.py             LLMClient ABC + AnthropicClient, OpenAIClient, GeminiClient + get_llm
  db.py              asyncpg pool init, get_book, set_status, save_summary
  pipeline/
    __init__.py
    runner.py        orchestrator: drives one book end to end, owns status transitions
    summarize.py     LLM calls for whole-book and per-chapter summaries (name_only path)
    ingest.py        [NEXT] PDF parse and chapter detection (pdf path)
    embed.py         [V1.2] embed chapter summaries into pgvector
    research.py      [V1.1] gather sourced critique and support
```

---

## Tech decisions (do not change without discussion)

### One worker process, async

The worker is a single `asyncio` process using `BRPOP`. Do not add threading,
multiprocessing, or concurrent job execution at this stage.

### asyncpg directly, no ORM

Same as the backend: raw SQL, transparent, fast. All DB operations in `app/db.py`.
Do not introduce SQLAlchemy, Tortoise, or any other ORM.

### Lazy SDK imports

Each LLM client imports its SDK inside `__init__` so the package only needs to
be installed for the provider in use. Keep it this way. Do not move SDK imports
to the module level.

### Idempotent pipeline

`save_summary` deletes existing chapters before inserting new ones. Re-running
a book does not create duplicate rows. Keep all pipeline stages idempotent.

### JSON responses from the LLM

The summarize stage asks the LLM to respond in JSON only, with no markdown fences
or preamble. `_strip_fences` in `summarize.py` handles the case where a model
adds fences anyway. Always parse LLM responses with `_parse_json` rather than
`json.loads` directly.

### No full book text in the database

Even when a PDF is uploaded, the raw text is processed in memory (chunked and
summarized) and discarded. Only derived content (summaries) is persisted. The
PDF file path is stored in `books.source_ref` but the content is never stored.

---

## Conventions

- No double dashes anywhere: not in Python comments, not in SQL, not in strings.
  Use `#` for Python comments and block comments for SQL.
- No em dashes in any output or generated text.
- All timestamps are `timestamptz`, never plain `timestamp`.
- snake_case for all Python identifiers and SQL names.
- LLM prompts are defined as module-level private functions (prefixed `_`) in
  each pipeline stage file, not inline in the calling code.
- Keep system prompts and user prompts separate. Never merge them into one string.

---

## What NOT to do

- Do not store full book text in the database.
- Do not store or log API keys.
- Do not add concurrency to the worker without discussion.
- Do not move SDK imports to module level (keep them lazy).
- Do not add an ORM.
- Do not set `books.status` to `published`. That is the backend's job.
- Do not add the V1.1 research stage or V1.2 embed stage until those phases start.
- Do not use double dashes in comments or strings.
- Do not hardcode model names or prompt text in `runner.py`. They belong in the
  relevant stage file (`summarize.py`, `research.py`, etc.).
- Do not let a single failed book crash the worker process. Catch, log, set
  status to `failed`, and continue.

---

## Pipeline stages: built vs next

### Built (V1, name_only path)

`pipeline/summarize.py`
- `summarize_from_knowledge(llm, title, author, max_tokens) -> SummaryResult`
- Two LLM calls: one for whole-book (one_paragraph_summary + full_summary),
  one for chapter list with summaries.
- Prompts ask for JSON only. `_strip_fences` and `_parse_json` handle model
  quirks. Both prompts are private functions in this file.
- Returns a `SummaryResult` dataclass with `one_paragraph_summary`, `full_summary`,
  and a list of `ChapterSummary` dataclasses.

`pipeline/runner.py`
- `run_pipeline(pool, settings, book_id)` is the single entry point.
- Reads the book row, branches on `source_type`.
- `name_only`: calls `summarize_from_knowledge`, saves, sets `ready`.
- `pdf` / `url`: raises `NotImplementedError` (ingest stage not built yet).
- Any exception: sets status to `failed` and re-raises (worker catches and logs).

`main.py`
- Blocking `BRPOP` loop on `settings.job_queue`.
- Deserializes `{book_id}` from the payload.
- Calls `run_pipeline` and logs outcome.
- Catches all exceptions so the worker never crashes on a bad job.

### Next (V1, pdf path)

`pipeline/ingest.py`
- Accept a PDF file path from `books.source_ref`.
- Parse with `pypdf` (already in requirements.txt).
- Detect chapter boundaries from headings or page structure.
- Return a list of `{chapter_number, chapter_title, raw_text}` dicts.
- Fallback: if chapter detection fails, return the whole text as one chapter
  so the pipeline degrades gracefully rather than failing the book.
- Feed into a hierarchical summarize call (summarize each chapter's raw text,
  then summarize those summaries into the whole-book summary).

### V1.1 (sourced critique and support)

`pipeline/research.py`
- Web search for reviews, academic critiques, and supporting articles.
- Synthesize search results into paraphrased insights with attribution.
- Flag any insight that is about a living person with `about_living_person = true`.
- Write rows to the `sources` table (defined in the backend repo's V1.1 schema).
- Never reproduce quoted text from sources. Paraphrase and attribute only.

### V1.2 (embeddings for RAG chat)

`pipeline/embed.py`
- Embed each chapter summary using an embedding model (dimension 1536 to match
  the vector column in the chapters table).
- Write the embedding and model name to `chapters.embedding` and `chapters.model`.
- Registering the pgvector type on the connection is handled in `app/db.py`
  via `pgvector.asyncpg.register_vector`.

---

## Adding a new LLM provider (checklist)

1. Add a new class in `app/llm.py` inheriting `LLMClient`.
2. Implement `async def complete(self, system, user, max_tokens) -> str`.
3. Keep the SDK import lazy (inside `__init__`).
4. Add a branch in `get_llm` for the new provider name.
5. Add `new_provider_api_key: str | None = None` to `Settings` in `config.py`.
6. Add the key and a model example to `.env.example`.
7. Add the SDK package to `requirements.txt` with a comment.
