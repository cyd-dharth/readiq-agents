from __future__ import annotations

import abc

from app.config import Settings


class EmbeddingClient(abc.ABC):
    """Provider-agnostic interface for turning text into vector embeddings."""

    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string and return its vector."""
        ...

    @abc.abstractmethod
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings and return one vector per input, in order."""
        ...


class GeminiEmbeddingClient(EmbeddingClient):
    """Embedding client backed by the google-genai SDK."""

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        """Create the lazily-imported genai client for the given model and output dimensionality."""
        from google import genai  # lazy: only needed when gemini is the embedding provider

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        """Embed one string via embed_content, requesting the configured output dimensionality."""
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
        """Fan out individual embed calls concurrently, since this SDK has no native batch endpoint."""
        import asyncio

        results = await asyncio.gather(*[self.embed(t) for t in texts])
        return list(results)


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding client backed by the openai SDK."""

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        """Create the lazily-imported AsyncOpenAI client for the given model and output dimensionality."""
        from openai import AsyncOpenAI  # lazy: only needed when openai is the embedding provider

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        """Embed one string via the embeddings endpoint."""
        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )
        return resp.data[0].embedding

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings in one API call, since OpenAI's endpoint natively accepts a list."""
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [d.embedding for d in resp.data]


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    """Build the configured EmbeddingClient from settings, raising if its API key is missing."""
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
