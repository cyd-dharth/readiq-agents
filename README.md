# Book summary agents

The generation service. It consumes a job per book from Redis, runs the pipeline,
and writes the results back to the same Postgres database the backend reads from.

## Stack

- Redis worker (consumes jobs the backend enqueues)
- asyncpg for writing summaries and chapters
- A provider-agnostic LLM client: Anthropic, OpenAI, or Gemini, chosen by env

## Prerequisites

Postgres and Redis. The backend repo's `docker compose up` brings both up; this
service just points at them via `DATABASE_URL` and `REDIS_URL`.

## Run

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.worker
```

Set `LLM_PROVIDER` and the matching key in `.env`. You only need the SDK for the
provider you use; the imports are lazy.

## Pipeline

Implemented:

- `pipeline/summarize.py`  whole-book and per-chapter summaries (name_only path)
- `pipeline/runner.py`     orchestration and status transitions
- `worker.py`              the Redis consumer loop

Next:

- `pipeline/ingest.py`     PDF parse and chapter detection
- `pipeline/embed.py`      chapter-summary embeddings into pgvector
- `pipeline/research.py`   (V1.1) critique and support with sources
