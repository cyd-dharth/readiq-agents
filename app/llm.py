from __future__ import annotations

import abc
import json

from pydantic import BaseModel

from app.config import Settings


class LLMClient(abc.ABC):
    """A minimal provider-agnostic text completion interface."""

    @abc.abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        ...

    @abc.abstractmethod
    async def complete_json(self, system: str, user: str, max_tokens: int, schema: type[BaseModel]) -> dict:
        """Call the LLM and return a dict guaranteed to match schema.

        Uses each provider's native structured-output API so the response is
        valid JSON by construction, not by prompt engineering.
        """
        ...


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic  # lazy: only used providers need installing

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    async def complete_json(self, system: str, user: str, max_tokens: int, schema: type[BaseModel]) -> dict:
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
    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
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


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        from google.genai import types

        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text or ""

    async def complete_json(self, system: str, user: str, max_tokens: int, schema: type[BaseModel]) -> dict:
        from google.genai import types

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


def get_llm(settings: Settings) -> LLMClient:
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
        return GeminiClient(settings.gemini_api_key, settings.llm_model)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
