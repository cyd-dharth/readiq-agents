from __future__ import annotations

import abc
import asyncio
import json
import re
import time
from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel

from app.config import Settings

_T = TypeVar("_T")
_GEMINI_DEFAULT_BACKOFF_SECONDS = 5.0


class _RateLimiter:
    """Paces calls to at most calls_per_minute, project-wide.

    Shared across all concurrent callers via a lock, so a burst of tasks
    queues up and drains at the configured rate instead of firing at once.
    """

    def __init__(self, calls_per_minute: int) -> None:
        self._min_interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_allowed_at: float | None = None

    async def wait_for_turn(self) -> None:
        """Block until the next call is allowed under the configured rate, if pacing is enabled."""
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if self._next_allowed_at is not None and now < self._next_allowed_at:
                await asyncio.sleep(self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at or 0) + self._min_interval


class LLMClient(abc.ABC):
    """A minimal provider-agnostic text completion interface."""

    @abc.abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return a freeform text completion for the given system and user prompts."""
        ...

    @abc.abstractmethod
    async def complete_json(self, system: str, user: str, max_tokens: int, schema: type[BaseModel]) -> dict:
        """Call the LLM and return a dict guaranteed to match schema.

        Uses each provider's native structured-output API so the response is
        valid JSON by construction, not by prompt engineering.
        """
        ...


class AnthropicClient(LLMClient):
    """LLMClient backed by Anthropic's Messages API (anthropic.AsyncAnthropic)."""

    def __init__(self, api_key: str, model: str) -> None:
        """Construct the Anthropic SDK client, importing the SDK lazily."""
        from anthropic import AsyncAnthropic  # lazy: only used providers need installing

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Call messages.create and concatenate the response's text blocks."""
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    async def complete_json(self, system: str, user: str, max_tokens: int, schema: type[BaseModel]) -> dict:
        """Force tool use with a synthetic structured_output tool so Anthropic returns schema-matching JSON."""
        # Force tool use so the model must return JSON matching the schema.
        tool_name = "structured_output"
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{
                "name": tool_name,
                "description": "Return the structured result.",
                "input_schema": schema.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": tool_name},
        )
        from anthropic.types import ToolUseBlock
        for block in resp.content:
            if isinstance(block, ToolUseBlock):
                return block.input  # type: ignore[return-value]
        raise RuntimeError("Anthropic returned no tool_use block")


class OpenAIClient(LLMClient):
    """LLMClient backed by OpenAI's chat completions API (openai.AsyncOpenAI)."""

    def __init__(self, api_key: str, model: str) -> None:
        """Construct the OpenAI SDK client, importing the SDK lazily."""
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Call chat.completions.create and return the message content."""
        resp = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    async def complete_json(self, system: str, user: str, max_tokens: int, schema: type[BaseModel]) -> dict:
        """Use beta.chat.completions.parse with response_format=schema for native structured output."""
        resp = await self._client.beta.chat.completions.parse(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed structured output")
        return parsed.model_dump()


def _gemini_retry_delay_seconds(exc: Exception) -> float:
    """Parse the wait time Gemini suggests in a 429 error message, falling back to a default backoff."""
    # The API embeds the suggested wait in the error message, e.g. "Please retry in 17.03s".
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
    if match:
        return float(match.group(1))
    return _GEMINI_DEFAULT_BACKOFF_SECONDS


async def _with_gemini_retry(call: Callable[[], Awaitable[_T]], max_retries: int) -> _T:
    """Retry call on a 429 ClientError, sleeping for the API's suggested delay each time.

    Any other error, or exhausting max_retries, is re-raised immediately.
    """
    from google.genai.errors import ClientError

    for attempt in range(max_retries):
        try:
            return await call()
        except ClientError as exc:
            if getattr(exc, "code", None) != 429 or attempt == max_retries - 1:
                raise
            await asyncio.sleep(_gemini_retry_delay_seconds(exc))
    raise AssertionError("unreachable")  # loop always returns or raises


class GeminiClient(LLMClient):
    """LLMClient backed by google.genai. Adds 429 retry with backoff and a shared per-minute rate limiter."""

    def __init__(self, api_key: str, model: str, max_retries: int, rate_limit_per_minute: int) -> None:
        """Construct the Gemini SDK client and its retry/rate-limiter settings, importing the SDK lazily."""
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._rate_limiter = _RateLimiter(rate_limit_per_minute)

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Call generate_content under rate limiting and 429 retry, returning the response text."""
        from google.genai import types

        async def call() -> str:
            await self._rate_limiter.wait_for_turn()
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text or ""

        return await _with_gemini_retry(call, self._max_retries)

    async def complete_json(self, system: str, user: str, max_tokens: int, schema: type[BaseModel]) -> dict:
        """Call generate_content with response_schema for native JSON output, under rate limiting and retry."""
        from google.genai import types

        async def call() -> dict:
            await self._rate_limiter.wait_for_turn()
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            if resp.text is None:
                raise RuntimeError("Gemini returned empty response")
            return json.loads(resp.text)

        return await _with_gemini_retry(call, self._max_retries)


def get_llm(settings: Settings) -> LLMClient:
    """Factory: build the LLMClient selected by settings.llm_provider, guarding on the matching API key."""
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return AnthropicClient(settings.anthropic_api_key, settings.llm_model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIClient(settings.openai_api_key, settings.llm_model)
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return GeminiClient(
            settings.gemini_api_key,
            settings.llm_model,
            settings.gemini_max_retries,
            settings.gemini_rate_limit_per_minute,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
