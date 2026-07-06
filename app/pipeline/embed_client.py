from __future__ import annotations

import abc

from app.config import Settings


class EmbeddingClient(abc.ABC):
    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abc.abstractmethod
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        from google import genai  # lazy: only needed when gemini is the embedding provider

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        from google.genai import types

        result = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self._dimensions,
            ),
        )
        return result.embeddings[0].values

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        results = await asyncio.gather(*[self.embed(t) for t in texts])
        return list(results)


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        from openai import AsyncOpenAI  # lazy: only needed when openai is the embedding provider

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )
        return resp.data[0].embedding

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [d.embedding for d in resp.data]


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    provider = settings.embedding_provider.lower()
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return GeminiEmbeddingClient(
            settings.gemini_api_key,
            settings.embedding_model,
            settings.embedding_dimensions,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIEmbeddingClient(
            settings.openai_api_key,
            settings.embedding_model,
            settings.embedding_dimensions,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
