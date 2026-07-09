import asyncio
import json
import logging

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from app import db
from app.config import get_settings
from app.pipeline.runner import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agents.worker")


async def main() -> None:
    """Redis worker entry point. Blocks on BRPOP for one job at a time, runs the full
    pipeline for each book, and keeps looping (retrying on Redis errors, logging and
    skipping bad payloads or pipeline failures) until the process is stopped."""
    settings = get_settings()
    pool = await db.init_pool()
    client = redis.from_url(settings.redis_url, decode_responses=True)
    log.info("Worker started, waiting for jobs on '%s'", settings.job_queue)

    try:
        while True:
            try:
                item = await client.brpop(settings.job_queue, timeout=5)
            except RedisConnectionError as exc:
                log.warning("Redis unavailable, retrying in 5s: %s", exc)
                await asyncio.sleep(5)
                continue
            except RedisTimeoutError as exc:
                log.warning("Redis command timed out, retrying in 5s: %s", exc)
                await asyncio.sleep(5)
                continue

            if item is None:
                continue

            _, payload = item
            try:
                book_id = json.loads(payload)["book_id"]
            except (ValueError, KeyError) as exc:
                log.error("Skipping bad job payload %r: %s", payload, exc)
                continue

            log.info("Processing book %s", book_id)
            try:
                await run_pipeline(pool, settings, book_id)
                log.info("Finished book %s", book_id)
            except Exception:
                log.exception("Pipeline failed for book %s", book_id)
    finally:
        await pool.close()
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
