from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from app.llm import LLMClient
from app.pipeline.search import SearchClient, SearchResult

logger = logging.getLogger(__name__)

_CRITIQUE_SYSTEM = (
    "You are a careful research analyst finding intellectual opposition "
    "to a book's arguments. You write in your own words, never quoting "
    "source text. You return JSON only, no markdown fences."
)

_SUPPORT_SYSTEM = (
    "You are a careful research analyst finding books and articles that "
    "reinforce or extend a book's core arguments. You write in your own "
    "words, never quoting source text. You return JSON only, no markdown "
    "fences."
)


class SourceItem(BaseModel):
    stance: str
    source_type: str
    title: str
    author_or_outlet: str | None
    reference_url: str
    insight: str
    about_living_person: bool = False
    verified: bool = False
    relevance_score: int = 0
    quality_score: int = 0


class ResearchResult(BaseModel):
    critiques: list[SourceItem] = []
    supports: list[SourceItem] = []


class _EvaluatedItem(BaseModel):
    title: str
    url: str
    author_or_outlet: str | None = None
    source_type: str
    insight: str
    about_living_person: bool = False
    relevance_score: int
    quality_score: int


class _EvaluatedItems(BaseModel):
    items: list[_EvaluatedItem]


def _critique_search_queries(title: str, author: str | None) -> list[str]:
    queries = [
        f"{title} {author or ''} criticism".strip(),
        f"{title} {author or ''} critique academic".strip(),
    ]
    if author:
        queries.append(f"{author} wrong oversimplification")
    queries.append(f"{title} counterargument rebuttal")
    return queries


def _support_search_queries(title: str, author: str | None) -> list[str]:
    queries = [
        f"books similar to {title} {author or ''}".strip(),
        f"{title} {author or ''} related works".strip(),
        f"{title} {author or ''} recommended reading".strip(),
    ]
    if author:
        queries.append(f"books influenced by {title}")
    else:
        queries.append(f"books like {title}")
    return queries


def _evaluate_prompt(
    title: str,
    author: str | None,
    summary: str,
    stance: str,
    results: list[SearchResult],
) -> str:
    by = f" by {author}" if author else ""
    results_block = "\n\n".join(
        f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}" for r in results
    )
    return (
        f"Book: '{title}'{by}\n"
        f"One paragraph summary: {summary}\n"
        f"Stance to evaluate for: {stance}\n\n"
        "Here are web search results:\n\n"
        f"{results_block}\n\n"
        "Evaluate each result and return the qualifying ones in the 'items' array. "
        "Each item needs: title, url, author_or_outlet (or null), "
        "source_type ('book', 'article', or 'academic_paper'), "
        "insight (2 to 4 sentences in your own words), "
        "about_living_person (bool), relevance_score (1 to 3), "
        "quality_score (1 to 3).\n\n"
        "Rules:\n"
        "Only include items with a real URL from the search results above. "
        "Never invent URLs.\n"
        "Discard Amazon reviews, Goodreads ratings, and listicles.\n"
        "Discard items where the insight cannot be written in your own words.\n"
        "relevance_score: 3 means it directly engages with the book's arguments, "
        "2 means related topic, 1 means tangential.\n"
        "quality_score: 3 means academic or major publication, 2 means reputable "
        "outlet, 1 means low quality blog or forum.\n"
        "Return an empty items array if nothing meets the bar.\n"
        "Return JSON only."
    )


def _dedupe_by_url(results: list[SearchResult], seen_urls: set[str]) -> list[SearchResult]:
    unique = []
    for r in results:
        if r.url in seen_urls:
            continue
        seen_urls.add(r.url)
        unique.append(r)
    return unique


async def _search_all(search: SearchClient, queries: list[str]) -> list[SearchResult]:
    results_per_query = await asyncio.gather(*(search.search(q, max_results=5) for q in queries))
    flat = [r for results in results_per_query for r in results]
    return flat


async def _evaluate_results(
    llm: LLMClient,
    title: str,
    author: str | None,
    summary: str,
    stance: str,
    results: list[SearchResult],
    max_tokens: int,
) -> list[_EvaluatedItem]:
    if not results:
        return []
    evaluated = _EvaluatedItems.model_validate(
        await llm.complete_json(
            _CRITIQUE_SYSTEM if stance == "critique" else _SUPPORT_SYSTEM,
            _evaluate_prompt(title, author, summary, stance, results),
            max_tokens,
            _EvaluatedItems,
        )
    )
    return evaluated.items


def _passes_filter(item: _EvaluatedItem) -> bool:
    return item.relevance_score >= 2 and item.quality_score >= 2


def _to_source_item(item: _EvaluatedItem, stance: str) -> SourceItem:
    return SourceItem(
        stance=stance,
        source_type=item.source_type,
        title=item.title,
        author_or_outlet=item.author_or_outlet,
        reference_url=item.url,
        insight=item.insight,
        about_living_person=item.about_living_person,
        verified=False,
        relevance_score=item.relevance_score,
        quality_score=item.quality_score,
    )


async def _run_stance_agent(
    llm: LLMClient,
    search: SearchClient,
    title: str,
    author: str | None,
    summary: str,
    max_tokens: int,
    min_items: int,
    max_items: int,
    stance: str,
    queries: list[str],
    retry_queries: list[str],
) -> list[SourceItem]:
    seen_urls: set[str] = set()

    results = await _search_all(search, queries)
    unique_results = _dedupe_by_url(results, seen_urls)

    evaluated = await _evaluate_results(llm, title, author, summary, stance, unique_results, max_tokens)
    qualifying = [item for item in evaluated if _passes_filter(item)]

    logger.info(
        "%s agent: %d searches run, %d results found, %d items after filter",
        stance, len(queries), len(unique_results), len(qualifying),
    )

    if len(qualifying) < min_items:
        retry_results = await _search_all(search, retry_queries)
        unique_retry_results = _dedupe_by_url(retry_results, seen_urls)
        retry_evaluated = await _evaluate_results(
            llm, title, author, summary, stance, unique_retry_results, max_tokens
        )
        retry_qualifying = [item for item in retry_evaluated if _passes_filter(item)]
        qualifying.extend(retry_qualifying)
        logger.info(
            "%s agent: retry ran %d searches, %d results found, %d additional items after filter",
            stance, len(retry_queries), len(unique_retry_results), len(retry_qualifying),
        )

    qualifying.sort(key=lambda item: item.relevance_score + item.quality_score, reverse=True)
    top_items = qualifying[:max_items]
    return [_to_source_item(item, stance) for item in top_items]


async def _run_critique_agent(
    llm: LLMClient,
    search: SearchClient,
    title: str,
    author: str | None,
    summary: str,
    max_tokens: int,
    min_items: int,
    max_items: int,
) -> list[SourceItem]:
    retry_queries = ["{0} academic criticism philosophy".format(title)]
    if author:
        retry_queries.append(f"{author} critics scholars debate")
    return await _run_stance_agent(
        llm, search, title, author, summary, max_tokens, min_items, max_items,
        stance="critique",
        queries=_critique_search_queries(title, author),
        retry_queries=retry_queries,
    )


async def _run_support_agent(
    llm: LLMClient,
    search: SearchClient,
    title: str,
    author: str | None,
    summary: str,
    max_tokens: int,
    min_items: int,
    max_items: int,
) -> list[SourceItem]:
    retry_queries = [f"books extending {title} arguments"]
    if author:
        retry_queries.append(f"academic works agreeing with {author}")
    return await _run_stance_agent(
        llm, search, title, author, summary, max_tokens, min_items, max_items,
        stance="support",
        queries=_support_search_queries(title, author),
        retry_queries=retry_queries,
    )


async def run_research(
    llm: LLMClient,
    search: SearchClient,
    title: str,
    author: str | None,
    summary: str,
    max_tokens: int,
    min_items: int,
    max_items: int,
) -> ResearchResult:
    async def safe_run(agent_name: str, coro) -> list[SourceItem]:
        try:
            return await coro
        except Exception:
            logger.warning("%s agent failed, returning no items for this stance", agent_name, exc_info=True)
            return []

    critiques, supports = await asyncio.gather(
        safe_run(
            "critique",
            _run_critique_agent(llm, search, title, author, summary, max_tokens, min_items, max_items),
        ),
        safe_run(
            "support",
            _run_support_agent(llm, search, title, author, summary, max_tokens, min_items, max_items),
        ),
    )
    return ResearchResult(critiques=critiques, supports=supports)
