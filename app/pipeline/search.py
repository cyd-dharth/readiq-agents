from __future__ import annotations

import abc

from pydantic import BaseModel

from app.config import Settings


class SearchResult(BaseModel):
    """One web search hit: title, URL, snippet, and optional source label."""

    title: str
    url: str
    snippet: str
    source: str | None = None


class SearchClient(abc.ABC):
    """Provider-agnostic interface for running a web search."""

    @abc.abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a single search query and return up to max_results results."""
        ...


class TavilyClient(SearchClient):
    """Search client backed by the Tavily API."""

    def __init__(self, api_key: str) -> None:
        """Create the lazily-imported AsyncTavilyClient."""
        from tavily import AsyncTavilyClient  # lazy: only needed when tavily is the search provider

        self._client = AsyncTavilyClient(api_key=api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search via Tavily, skipping any result with an empty URL and never inventing one."""
        response = await self._client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
        )
        results = []
        for r in response.get("results", []):
            url = r.get("url", "").strip()
            if not url:
                continue
            results.append(SearchResult(
                title=r.get("title", "").strip(),
                url=url,
                snippet=r.get("content", "").strip(),
                source=r.get("source"),
            ))
        return results


def get_search_client(settings: Settings) -> SearchClient:
    """Build the configured SearchClient from settings, raising if its API key is missing."""
    provider = settings.search_provider.lower()
    if provider == "tavily":
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        return TavilyClient(settings.tavily_api_key)
    raise ValueError(f"Unknown search provider: {settings.search_provider}")
