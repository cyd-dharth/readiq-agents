# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file is the full context for this repo. Read it completely before writing
any code, suggesting changes, or answering questions. Every decision here was
made deliberately. Do not suggest alternatives unless asked.

---

## What this service is

The AI generation pipeline for a book summary platform. It has two entry points:

- A worker process (`run_agent.py`) that consumes jobs from a Redis queue, runs the
  full pipeline for each book, and writes results back to the shared Postgres
  database.
- An internal HTTP API (`run_api.py` / `app/server.py`) that serves RAG-backed
  chat answers about a book, called only by the backend service. Never exposed
  publicly.

This is one of three repos:

- `booksummary-backend`: FastAPI controller, public API, job dispatch, DB schema
- `booksummary-agents` (this repo): the AI generation pipeline and internal chat API
- `booksummary-frontend`: Astro website that renders published content

Infrastructure (Postgres and Redis) is brought up by the backend repo's
`docker-compose.yml`. This service connects to both via env vars.

---

## Project context

### What the platform does

Users visit a website showing AI-generated book summaries. Each book page has a
one-paragraph TL;DR, a full summary, chapter-by-chapter summaries, sourced
critique and support with reference links, and a RAG-powered chat. The pipeline
runs locally; the live site only reads published rows.

### Phasing

- V1 (done): whole-book summary, chapter summaries, status written to Postgres.
  Covers `name_only`, `pdf`, and `epub` source types.
- V1.1 (done): sourced critique and support with reference links, stances, and
  a verified flag, written to the `sources` table.
- V1.2 (done): chapter-summary embeddings written to pgvector, plus an internal
  chat HTTP API that performs retrieval and answers questions grounded in a
  book's summaries and sources.
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
# Set TAVILY_API_KEY for the research stage
# Set EMBEDDING_PROVIDER / EMBEDDING_MODEL for the embed stage

# 4. Run the worker (Redis consumer, runs the pipeline)
uv run python run_agent.py

# 5. Run the internal chat API (separate process, separate port)
uv run python run_api.py
```

Dependencies are declared in `pyproject.toml` / `uv.lock`, which is what `uv`
actually installs from. `requirements.txt` is kept in sync manually as a
secondary reference; when adding a dependency, use `uv add <package>` so both
stay consistent.

There are no automated tests yet. There is no linter configured. The worker
logs every job it picks up, its outcome, and any exceptions with full
tracebacks. The chat API logs every request failure with a full traceback.

A `Dockerfile` is present for containerized deployment; it is not used in the
local development flow.

---

## Architecture

```
Redis queue  <--  backend enqueues {book_id}
     |
     v
run_agent.py  (BRPOP loop, one job at a time)
     |
     v
pipeline/runner.py  (orchestrates stages, owns status transitions)
     |
     |-- pipeline/ingest.py       (PDF parse + 4-tier chapter detection)
     |-- pipeline/ingest_epub.py  (EPUB parse + chapter/spine extraction)
     |-- pipeline/summarize.py    (LLM calls: whole-book + per-chapter, name_only/pdf/epub)
     |-- pipeline/embed.py        (embeds chapter summaries into pgvector)
     |-- pipeline/research.py     (critique + support sourcing via web search)
     |
     v
Postgres  (writes summaries, chapters, embeddings, sources; flips status)


run_api.py  (uvicorn entry point, separate process from the worker)
     |
     v
app/server.py  (FastAPI app, lifespan owns its own asyncpg pool)
     |
     v
POST /chat
     |
     |-- pipeline/retrieval.py  (question classification + vector search + sources fetch)
     |-- pipeline/chat.py       (system/user prompt building, calls LLMClient.complete)
     |
     v
Postgres  (reads book summary, chapters.embedding, sources; read-only)
```

The worker is a single async process. It blocks on Redis with `BRPOP`, picks up
one job, runs the full pipeline for that book, then blocks again. One book at a
time is correct for V1; concurrency can be added later if needed.

The chat API is a second, independent process with its own `asyncpg` pool
(created in `app/server.py`'s lifespan, not shared with `app/db.py`'s pool).
Do not merge the worker and the chat API into one process; they have different
lifecycles and failure domains.

---

## Database schema (read-only for this service)

This service does not own the schema. The DDL lives in the backend repo. This
service only reads and writes rows.

All queries execute within the `booksummary` Postgres schema, set via
`search_path` in the connection pool (controlled by `db_schema` in `config.py`).

Both connection pools (`app/db.py`'s worker pool and `app/server.py`'s chat API
pool) register the pgvector type via `pgvector.asyncpg.register_vector` in
their `init` callback. This is required for any query that binds a Python
`list[float]` to a `vector` column, or casts a bound parameter to `::vector`.
Without it, writes to `chapters.embedding` fail with `asyncpg.exceptions.DataError`.

### books (relevant columns)

- `id` uuid PK
- `title`, `author`
- `source_type`: `name_only` | `pdf` | `epub` | `url`
- `source_ref`: path to uploaded file, NULL for name_only
- `one_paragraph_summary`: written by this service after generation
- `full_summary`: written by this service after generation
- `status`: this service reads `processing`/`ready`/`published` and writes
  `processing`, `ready`, or `failed`
- `research_status`: `pending` | `completed` | `failed` | `skipped`, owned by
  this service's research stage (see Status transitions below)
- `copyright_status`: `public_domain` | `in_copyright` | `permission_granted`

### chapters (written by this service)

- `id` uuid PK, `book_id` FK (cascade delete)
- `chapter_number`, `chapter_title`, `summary`, `word_count`
- `embedding` vector(1536): written by the embed stage, NULL until then
- `model`: embedding model id, for re-embedding on model change
- UNIQUE on (book_id, chapter_number)

For `pdf` and `epub` books, chapter rows are created as stubs (title and
number, `summary = NULL`) immediately after ingest, before any chapter is
summarized. `summarize_from_chapters` then fills in `summary` one chapter (or
one batch) at a time. This makes the pipeline resumable: if the worker crashes
mid-book, re-running it only summarizes chapters where `summary IS NULL`
instead of redoing the whole book. Stub inserts use
`ON CONFLICT (book_id, chapter_number) DO NOTHING` so a resumed run never wipes
an already-written summary.

For `name_only` books, `save_summary` still deletes and reinserts all chapters
in one shot, since there is no incremental chapter generation to resume.

### sources (written by the research stage)

- `book_id` FK
- `stance`: `critique` | `support`
- `source_type`: `book` | `article` | `academic_paper`
- `title`, `author_or_outlet` (nullable)
- `reference_url`: always a real URL from a search result, never invented
- `insight`: 2 to 4 sentences, paraphrased, never quoted from the source
- `about_living_person`: bool
- `verified`: bool, always `False` from the pipeline today (no human review step yet)

`save_sources` deletes all existing sources for a book before inserting the
new batch, so re-running research is idempotent. Inserts use
`asyncpg`'s `executemany`.

---

## Status transitions owned by this service

### books.status

```
processing  ->  ready    (summarization completed successfully)
processing  ->  failed   (any unhandled exception during summarization)
```

The backend owns all other transitions, including `ready -> published`. This
service never sets `books.status` to `published`.

### books.research_status

```
pending  ->  completed   (research stage found and wrote sources, or found none)
pending  ->  failed      (research stage raised an unhandled exception)
```

Research and embedding run only after `books.status` is already `ready` (or,
for the retrigger fast path below, was already `ready` from a prior run).
A failure in either stage is caught, logged, and never flips `books.status`
back to `failed`. `research_status` is the only thing that reflects a research
failure; the book itself stays usable.

### Retriggering an already-processed book

`run_pipeline` checks `books.status` before doing any work:

- `status` in (`ready`, `published`) and `research_status == "completed"`:
  logs and returns immediately. True no-op, nothing is re-run.
- `status` in (`ready`, `published`) and `research_status` in (`pending`,
  `failed`, `skipped`): skips ingest and summarization entirely (they are
  assumed already done) and runs only the embed and research stages.
- Otherwise (`status` is `processing` or `failed`): runs the full pipeline
  from scratch, as before. Chapter-level resume (see above) still applies
  within this path.

This means retriggering a fully-published book with `research_status=pending`
(for example, right after the `research_status` column was added and
backfilled to `pending` for existing rows) will only run research and embed,
not redo any summarization.

On failure during summarization, the exception is logged with full traceback
and `books.status` is set to `failed`. The worker does not crash; it logs and
moves on to the next job.

---

## LLM client

A provider-agnostic interface in `app/llm.py`. One abstract base class
(`LLMClient`) with two methods:

- `complete(system, user, max_tokens) -> str`: freeform text completion.
- `complete_json(system, user, max_tokens, schema: type[BaseModel]) -> dict`:
  structured output. Uses each provider's native structured-output/tool-use
  API so the response is valid JSON by construction, not by prompt engineering
  or manual fence-stripping. This is the only way LLM JSON responses are
  parsed anywhere in this codebase; there is no `_parse_json` or
  `_strip_fences` helper.

Three implementations behind a factory function `get_llm(settings)`:

- `AnthropicClient`: uses `anthropic.AsyncAnthropic`, `messages.create`.
  `complete_json` forces tool use with a synthetic `structured_output` tool
  whose `input_schema` is the pydantic schema's JSON schema.
- `OpenAIClient`: uses `openai.AsyncOpenAI`, `chat.completions.create` for
  `complete`, `beta.chat.completions.parse` with `response_format=schema` for
  `complete_json`.
- `GeminiClient`: uses `google.genai.Client`, `client.aio.models.generate_content`
  with `types.GenerateContentConfig(system_instruction=..., max_output_tokens=...)`.
  For `complete_json`, adds `response_mime_type="application/json"` and
  `response_schema=schema`. Uses the current `google-genai` SDK, not the
  deprecated `google-generativeai`.

The provider is selected by `LLM_PROVIDER` in env. SDK imports are lazy (inside
`__init__` or the calling method) so only the installed provider's package is
needed. All three guard against a missing API key with a clear `RuntimeError`
at client construction time.

To add a new provider: create a new class inheriting `LLMClient`, implement
both `complete` and `complete_json`, add a branch in `get_llm`, add the key to
`Settings`, and add it to `.env.example`. Do not change the `LLMClient`
interface.

### Gemini-specific resilience

`GeminiClient` wraps every call in retry-with-backoff (`_with_gemini_retry`)
because the free tier's per-minute quota is easy to exceed:

- `gemini_max_retries` (`GEMINI_MAX_RETRIES`, default 10): on a `ClientError`
  with `code == 429`, sleeps for the delay the API suggests in its error
  message (parsed via regex, `_gemini_retry_delay_seconds`) and retries. Any
  other error, or exhausting retries, re-raises immediately.
- `gemini_rate_limit_per_minute` (`GEMINI_RATE_LIMIT_PER_MINUTE`, default 5):
  a shared `_RateLimiter` inside `GeminiClient` paces every call (both
  `complete` and `complete_json`) to at most N per minute project-wide, so
  concurrent callers queue and drain at the configured rate instead of
  bursting and immediately hitting 429s. Set to `0` to disable pacing
  entirely (e.g. on a paid plan with a much higher quota).

On a paid Gemini plan: raise `GEMINI_RATE_LIMIT_PER_MINUTE` (or set to `0`),
lower `GEMINI_MAX_RETRIES`, and raise `MAX_CONCURRENT_CHAPTER_SUMMARIES` for
full speed. No code changes needed, only env vars.

---

## Embedding client

A second provider-agnostic interface, `app/pipeline/embed_client.py`, mirrors
`llm.py`'s pattern but for embeddings:

- `EmbeddingClient` ABC: `embed(text) -> list[float]`,
  `embed_many(texts) -> list[list[float]]`.
- `GeminiEmbeddingClient`: uses `google.genai`,
  `client.aio.models.embed_content` with `types.EmbedContentConfig(output_dimensionality=...)`.
  `embed_many` fans out individual `embed` calls via `asyncio.gather` (Gemini
  has no native batch embed endpoint in this SDK version).
- `OpenAIEmbeddingClient`: uses `openai.AsyncOpenAI`, `embeddings.create`,
  which natively accepts a list for batch embedding.
- `get_embedding_client(settings)`: factory, same key-presence guard pattern
  as `get_llm`.

Selected by `EMBEDDING_PROVIDER` (default `gemini`). The default
`EMBEDDING_MODEL` is `gemini-embedding-2`. Do not use `text-embedding-004`; it
returns 404 against the current Gemini API (deprecated/renamed). Confirmed
working Gemini embedding models as of this writing: `gemini-embedding-001`,
`gemini-embedding-2-preview`, `gemini-embedding-2`.
`EMBEDDING_DIMENSIONS` (default 1536) must match the `chapters.embedding`
column's vector dimension.

---

## Search client

A third provider-agnostic interface, `app/pipeline/search.py`, same pattern
again:

- `SearchClient` ABC: `search(query, max_results=5) -> list[SearchResult]`.
- `TavilyClient`: uses `tavily.AsyncTavilyClient`. Skips any result with an
  empty URL. Never fabricates a URL; if Tavily returns nothing, the caller
  gets an empty list.
- `get_search_client(settings)`: factory, guards on `TAVILY_API_KEY`.

Selected by `SEARCH_PROVIDER` (default and only current implementation:
`tavily`).

---

## File structure

```
run_agent.py                  Redis worker entry point: asyncio.run, BRPOP loop, job dispatch
run_api.py             chat API entry point: uvicorn.run("app.server:app", ...)
app/
  config.py              pydantic-settings, see Settings reference below
  llm.py                 LLMClient ABC + Anthropic/OpenAI/Gemini clients + get_llm + Gemini rate limiter
  db.py                  asyncpg pool init (worker pool, registers pgvector), all DB read/write functions
  chat_models.py         pydantic request/response models for the chat API
  server.py              FastAPI app: lifespan owns its own asyncpg pool, POST /chat, GET /health
  pipeline/
    __init__.py
    runner.py            orchestrator: drives one book end to end, owns status transitions
    ingest.py            PDF parse + 4-tier chapter detection (pdf path)
    ingest_epub.py        EPUB parse + chapter/spine extraction, non-chapter filtering (epub path)
    summarize.py          LLM calls for whole-book and per-chapter summaries; all three source types
    embed.py              embeds chapter summaries into pgvector
    embed_client.py        EmbeddingClient ABC + Gemini/OpenAI clients + get_embedding_client
    research.py            critique + support sourcing via web search, two concurrent sub-agents
    search.py              SearchClient ABC + TavilyClient + get_search_client
    retrieval.py            question-type classification, vector search, sources fetch for chat
    chat.py                 system/user prompt building for chat, calls LLMClient.complete
```

---


## Tech decisions (do not change without discussion)

### One worker process, async

The worker is a single `asyncio` process using `BRPOP`. Do not add threading,
multiprocessing, or concurrent job execution at this stage.

### The chat API is a separate process from the worker

`run_api.py` / `app/server.py` runs independently of `run_agent.py`. They have
separate `asyncpg` pools, separate lifecycles, and can be started, stopped, or
restarted independently. Do not merge them into a single process or share a
pool between them.

### asyncpg directly, no ORM

Same as the backend: raw SQL, transparent, fast. All DB operations for the
worker live in `app/db.py`. The chat API queries directly via its own pool in
`app/server.py` / `app/pipeline/retrieval.py`. Do not introduce SQLAlchemy,
Tortoise, or any other ORM.

### Lazy SDK imports

Each LLM, embedding, and search client imports its SDK inside `__init__` (or,
for Gemini's per-call config types, inside the method) so the package only
needs to be installed for the provider in use. Keep it this way. Do not move
SDK imports to the module level.

### Idempotent pipeline

Re-running a book does not create duplicate rows anywhere:
- `chapters`: stub inserts use `ON CONFLICT DO NOTHING`; `save_summary` (name_only
  path) deletes and reinserts.
- `sources`: `save_sources` deletes all existing rows for the book before inserting.
- `chapters.embedding`: `embed_chapters` overwrites via `UPDATE`, safe to rerun.

Keep all pipeline stages idempotent.

### JSON responses from the LLM

Every LLM call that needs structured output uses `LLMClient.complete_json`
with a pydantic schema, which relies on the provider's native structured
output or tool-use API. There is no manual JSON parsing, no fence-stripping,
and no `json.loads` on a freeform `complete()` response anywhere in this
codebase. If you see a reference to `_parse_json` or `_strip_fences`
elsewhere (old docs, old comments), it is stale; those functions do not exist.

### No full book text in the database

Even when a PDF or EPUB is uploaded, the raw text is processed in memory
(parsed, chapter-split, summarized) and discarded. Only derived content
(summaries, embeddings, sourced insights) is persisted. The file path is
stored in `books.source_ref` but the content is never stored.

### Never invent URLs

`research.py`'s evaluate prompt explicitly instructs the LLM to only include
items with a real URL taken from the search results, never to invent one.
`sources.reference_url` should always be traceable back to an actual search
hit.

---

## Conventions

- No double dashes anywhere: not in Python comments, not in SQL, not in
  strings (including prompt text and section dividers inside prompts; use
  `===` or similar instead of `---`). Use `#` for Python comments and block
  comments for SQL.
- No em dashes in any output or generated text.
- All timestamps are `timestamptz`, never plain `timestamp`.
- snake_case for all Python identifiers and SQL names.
- LLM prompts are defined as module-level private functions (prefixed `_`) in
  each pipeline stage file, not inline in the calling code.
- Keep system prompts and user prompts separate. Never merge them into one string.
- Every "never raise" stage (`embed_chapters`, `retrieve`, `run_research` and
  its sub-agents) logs a warning or exception and returns a safe empty/default
  value instead of propagating. When adding a new such stage, make sure every
  internal `try` actually covers the full body, not just the first call that
  happens to be inside one.

---

## What NOT to do

- Do not store full book text in the database.
- Do not store or log API keys.
- Do not add concurrency to the worker without discussion.
- Do not move SDK imports to module level (keep them lazy).
- Do not add an ORM.
- Do not set `books.status` to `published`. That is the backend's job.
- Do not use double dashes in comments, strings, or prompt text.
- Do not hardcode model names or prompt text in `runner.py` or `server.py`.
  They belong in the relevant stage file (`summarize.py`, `research.py`,
  `chat.py`, etc.).
- Do not let a single failed book crash the worker process. Catch, log, set
  status to `failed`, and continue.
- Do not let a failed embed or research stage affect `books.status`. Only
  `research_status` (or a logged warning, for embed) reflects that failure.
- Do not name a new root-level module `app.py`. It collides with the `app/`
  package directory and breaks every `from app import ...` / `from app.config
  import ...` in the codebase (confirmed by testing it).
- Do not put request-derived data that is also stored in Postgres (book title,
  author, summaries) into an API request body when the row can be looked up by
  ID instead. `/chat` takes only `book_id`, `question`, and optional `history`;
  the book's context is always fetched server-side from the DB, never trusted
  from the caller, so it can never drift from what is actually stored.
- Do not add auth or rate limiting inside this service. Both are the backend's
  responsibility; this service is only ever called by the backend, never
  exposed publicly.

---

## Pipeline stages

### `pipeline/ingest.py` (pdf path)

`ingest_pdf(file_path, llm, max_tokens) -> list[dict]`, returns
`[{"chapter_number", "chapter_title", "raw_text"}, ...]`. Never raises.

Parses with `pypdf`. If total extracted text is under
`unreadable_pdf_min_chars` (default 500), assumes the PDF is image-based with
no OCR layer and returns the whole text as one chapter immediately.

Otherwise runs a four-tier chapter detection cascade, each tier only
attempted if the previous one found fewer than 2 boundaries:

1. **Tier 1**: find table-of-contents pages (`toc_scan_pages`, default 10),
   ask the LLM to extract `{chapter_title, toc_page}` pairs, then calibrate
   the offset between TOC page numbers and actual pypdf page indices (since
   front matter shifts printed page numbers). Excludes the TOC pages
   themselves from the calibration search so the calibration cannot lock onto
   the TOC page's own text.
2. **Tier 2**: if the TOC has no page numbers, ask the LLM to extract just the
   titles (scanning `toc_titles_scan_pages`, default 6, if no TOC pages were
   found), then heuristically match each page's first line against the title
   list.
3. **Tier 3**: pure regex heuristics (chapter/part/section keywords, roman
   numerals, digit headings, all-caps short lines), no LLM call.
4. **Tier 4**: LLM analyzes the first two lines of every page (sampling every
   other page if the input exceeds `tier4_page_lines_char_limit`, default
   4000 chars) and picks out chapter-start pages directly. If this also finds
   fewer than 2 boundaries, chapter detection is considered failed.

If all four tiers fail, or any exception occurs anywhere in detection, falls
back to the whole book as a single chapter with `chapter_title = None`. This
fallback wraps the entire detection block, not just individual tiers.

### `pipeline/ingest_epub.py` (epub path)

`ingest_epub(file_path) -> list[dict]`, same return shape as `ingest_pdf`.
Synchronous (no LLM call needed) and never raises.

Uses `ebooklib` to read spine items in reading order and `beautifulsoup4` to
convert each item's HTML to clean text. Builds a chapter title map by walking
the EPUB's TOC (`_extract_toc_titles`), falling back to the first `h1`/`h2`/`h3`
in the item's HTML when the TOC has no entry. Filters out known non-chapter
content (`_is_non_chapter`: preface, index, bibliography, copyright,
Gutenberg boilerplate, etc., matched by title substring or by being under
150 characters). Merges suspiciously short spine items (under 300 characters)
into the preceding chapter, since some EPUBs split one logical chapter across
multiple small HTML files.

Falls back to a single whole-book chapter if `ebooklib`/`beautifulsoup4` are
not installed, if the spine is empty, if parsing produces no text content, or
if total extracted text is under 500 characters (image-based/no text layer).

### `pipeline/summarize.py`

Three entry points, one per source type:

- `summarize_from_knowledge(llm, title, author, max_tokens) -> SummaryResult`:
  the `name_only` path. Two `complete_json` calls: one for the whole-book
  summary, one for the full chapter list with summaries (asks the LLM to
  generate chapters from its own knowledge of the book).
- `summarize_from_chapters(llm, chapters, max_tokens, max_concurrent_chapters,
  pool, book_id, sequential=True) -> SummaryResult`: the `pdf`/`epub` path.
  Takes chapters from `ingest_pdf`/`ingest_epub`. Checks the DB for chapters
  already summarized (`summary IS NOT NULL`) and skips them, so a resumed run
  only processes what's left. Two modes:
  - `sequential=True` (default): chapters are summarized one at a time, in
    ascending `chapter_number` order, each call receiving a rolling 2-3
    sentence digest of everything summarized so far (`_SequentialChapter`
    schema returns both `summary` and `updated_digest` in one `complete_json`
    call). On resume, the digest starts empty rather than being reconstructed
    from already-done chapters; continuity is imperfect across a resume but
    the cost of rebuilding it was judged not worth it.
  - `sequential=False`: the original concurrent path, chapters summarized in
    parallel via `asyncio.gather`, capped by an `asyncio.Semaphore` sized from
    `max_concurrent_chapter_summaries`.
  Either way, each chapter's summary is written to the DB immediately via
  `db.save_chapter_summary` as soon as it's produced, not batched at the end.
  After all chapters are summarized, one final `complete_json` call
  synthesizes the whole-book summary from the chapter summaries (never the
  raw text), written via `db.save_book_summary`.

### `pipeline/runner.py`

`run_pipeline(pool, settings, book_id)` is the single entry point. See
"Status transitions" above for the retrigger fast path.

For a fresh run: reads the book row, branches on `source_type`
(`name_only`/`pdf`/`epub`; `url` and anything else raises `NotImplementedError`),
runs ingest and summarization, sets `status = ready`. Then, outside the status
try/except (so their failure cannot flip status back to `failed`), runs the
embed stage and the research stage in sequence, each independently wrapped so
one failing does not prevent the other from running.

### `pipeline/embed.py` and `pipeline/embed_client.py` (V1.2)

`embed_chapters(pool, book_id, embedder, model_name) -> None`. Fetches every
chapter with a non-null, non-empty summary, embeds them all via
`embedder.embed_many`, and writes `embedding` + `model` back per chapter in
one transaction. Skips (logs a warning, returns) if there are no summarized
chapters yet. Never raises: any embedding API failure is caught, logged, and
the function returns without writing anything. Idempotent: reruns overwrite.

Requires the connection pool used to have `pgvector.asyncpg.register_vector`
applied; both `app/db.py`'s `init_pool` and `app/server.py`'s lifespan pool
do this.

### `pipeline/research.py` (V1.1)

`run_research(llm, search, title, author, summary, max_tokens, min_items,
max_items) -> ResearchResult`. Runs two independent sub-agents concurrently
via `asyncio.gather`, each wrapped so one failing returns an empty list for
that stance rather than aborting the other:

- `_run_critique_agent`: searches for intellectual opposition to the book's
  arguments (`_critique_search_queries`: `"{title} {author} criticism"`,
  `"... critique academic"`, `"{author} wrong oversimplification"`,
  `"{title} counterargument rebuttal"`).
- `_run_support_agent`: searches for related and reinforcing works
  (`_support_search_queries`: `"books similar to ..."`, `"... related works"`,
  `"... recommended reading"`, `"books influenced by/like {title}"`).

Both share `_run_stance_agent`: run all queries concurrently
(`asyncio.gather`, max 5 results each), deduplicate by URL, send unique
results to the LLM via `_evaluate_prompt` for scoring (`relevance_score` and
`quality_score`, 1 to 3 each; the LLM also writes the paraphrased `insight`
and detects `about_living_person` in this same call, no separate evaluator
call), filter to `relevance_score >= 2 and quality_score >= 2`. If the
qualifying count is under `min_items`, runs two more reformulated queries and
merges in any new qualifying items (deduplicated against already-seen URLs).
Sorts by combined score descending, truncates to `max_items`.

`run_research` itself never raises. Writes nothing to the DB directly; the
caller (`runner.py`) converts `ResearchResult.critiques + .supports` into
dicts and calls `db.save_sources`.

### `pipeline/search.py` and `pipeline/embed_client.py`

See "Search client" and "Embedding client" sections above.

### Internal chat API: `app/server.py`, `app/chat_models.py`, `pipeline/retrieval.py`, `pipeline/chat.py`

`POST /chat` request body is `{book_id: str, question: str, history: list[dict] = []}`
(`ChatRequest` in `app/chat_models.py`). It does **not** take book title,
author, or summaries in the payload; the server fetches those itself via
`db.get_book_summary(pool, book_id)` and builds `BookContext` from the DB row.
An invalid `book_id` (not a UUID) returns 400; a `book_id` with no matching
row returns 404.

`pipeline/retrieval.py`'s `retrieve(pool, embedder, book_id, question,
max_chapters) -> RetrievedContext` never raises:

- Classifies the question into `critique` / `support` / `summary` / `default`
  via keyword matching (`_question_type`). This classification is currently
  only used to decide whether to skip vector search (see below); it does not
  filter which sources are returned.
- For `summary`-type questions, skips vector search entirely (the caller
  already has the full book summary via `BookContext`).
- For all other question types, embeds the question and runs a pgvector
  cosine-distance search (`embedding <=> $1::vector`) against
  `chapters.embedding`, limited to `max_chapters` (default 3, from
  `settings.max_context_chapters`). Any failure here (embedding call,
  vector search query) is caught and logged; the function continues with an
  empty `chapter_chunks` rather than raising.
- Always fetches all `sources` rows for the book, split by stance, regardless
  of question type. Also wrapped in its own try/except.

`pipeline/chat.py`'s `answer_question(llm, question, book_ctx, retrieved,
history, max_tokens, max_history_messages=8) -> str` builds a system prompt
(`_build_system`, book title/author only) and a user prompt (`_build_user_prompt`,
book overview + retrieved chapter chunks + critique sources + support sources
+ last `max_history_messages` history turns + the question), then calls
`llm.complete`. The chat system prompt instructs the model to answer strictly
from the provided context, never reproduce large passages, attribute sourced
critiques/support by title and outlet, and say so clearly when the context is
insufficient rather than filling gaps with general knowledge.

---

## Settings reference (`app/config.py`)

Grouped roughly by when each group was added:

**Core**: `database_url`, `db_schema`, `redis_url`, `job_queue`

**LLM**: `llm_provider` (anthropic/openai/gemini), `llm_model`,
`max_output_tokens`, `anthropic_api_key`, `openai_api_key`, `gemini_api_key`

**Uploads**: `upload_dir`

**PDF ingest tuning**: `toc_scan_pages`, `toc_titles_scan_pages`,
`unreadable_pdf_min_chars`, `tier4_page_lines_char_limit`

**Gemini resilience**: `gemini_max_retries`, `max_concurrent_chapter_summaries`,
`gemini_rate_limit_per_minute`

**Research (V1.1)**: `tavily_api_key`, `search_provider`,
`min_sources_per_stance`, `max_sources_per_stance`

**Embedding (V1.2)**: `embedding_provider`, `embedding_model`
(default `gemini-embedding-2`, see the embedding client note on why not
`text-embedding-004`), `embedding_dimensions`

**Chat API (V1.2)**: `chat_api_host`, `chat_api_port`, `max_context_chapters`,
`max_history_messages`

All settings are also present in `.env.example` with comments; keep both in
sync when adding a new one.

---

## Adding a new LLM provider (checklist)

1. Add a new class in `app/llm.py` inheriting `LLMClient`.
2. Implement both `async def complete(self, system, user, max_tokens) -> str`
   and `async def complete_json(self, system, user, max_tokens, schema) -> dict`.
3. Keep the SDK import lazy (inside `__init__` or the method body).
4. Add a branch in `get_llm` for the new provider name.
5. Add `new_provider_api_key: str | None = None` to `Settings` in `config.py`.
6. Add the key and a model example to `.env.example`.
7. Add the SDK package via `uv add <package>` (updates `pyproject.toml` and
   `uv.lock`); mirror it into `requirements.txt` with a comment.

## Adding a new embedding or search provider

Same shape as above: new class inheriting `EmbeddingClient` (in
`embed_client.py`) or `SearchClient` (in `search.py`), lazy SDK import, a
branch in the relevant `get_*_client` factory, a new `*_api_key` setting, and
an `.env.example` entry.
