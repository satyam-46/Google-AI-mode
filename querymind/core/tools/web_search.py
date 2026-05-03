"""LangChain-compatible async web search tool."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient
from tavily.errors import MissingAPIKeyError, UsageLimitExceededError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

_LOG = logging.getLogger(__name__)
_CONCURRENCY = int(os.getenv("TAVILY_CONCURRENCY", "5"))
_semaphore = asyncio.Semaphore(_CONCURRENCY)


class SearchResult(BaseModel):
    """Normalized web search result returned by QueryMind."""

    url: str
    title: str
    content: str
    relevance_score: float = 0.0


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query to send to Tavily.")
    top_k: int = Field(5, ge=1, le=10, description="Maximum number of results to return.")


def _normalize_result(item: dict[str, Any]) -> SearchResult:
    return SearchResult(
        url=str(item.get("url") or ""),
        title=str(item.get("title") or ""),
        content=str(item.get("content") or item.get("snippet") or ""),
        relevance_score=float(item.get("score") or item.get("relevance_score") or 0.0),
    )


async def web_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """Search the web with Tavily and return normalized results.

    If `TAVILY_API_KEY` is not configured, this returns an empty list so the
    rest of the LangChain pipeline remains runnable during local learning.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        _LOG.info("TAVILY_API_KEY is not set; returning empty web results")
        return []

    async with _semaphore:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((TimeoutError, UsageLimitExceededError)),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                client = AsyncTavilyClient(api_key=api_key)
                response = await client.search(
                    query=query,
                    max_results=top_k,
                    search_depth="basic",
                    include_raw_content=False,
                )
                return [_normalize_result(item) for item in response.get("results", [])]

    return []


async def _web_search_tool(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    results = await web_search(query=query, top_k=top_k)
    return [result.model_dump() for result in results]


web_search_tool = StructuredTool.from_function(
    coroutine=_web_search_tool,
    name="web_search",
    description="Search the web for current evidence and return normalized results.",
    args_schema=WebSearchInput,
)


__all__ = ["SearchResult", "WebSearchInput", "web_search", "web_search_tool"]
