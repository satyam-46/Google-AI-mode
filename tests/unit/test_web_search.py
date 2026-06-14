import pytest
import importlib

from core.tools.web_search import SearchResult, web_search, web_search_tool

web_search_module = importlib.import_module("core.tools.web_search")


@pytest.mark.asyncio
async def test_web_search_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    results = await web_search("test query", top_k=2)

    assert results == []


@pytest.mark.asyncio
async def test_web_search_normalizes_tavily_results(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        async def search(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "content": "Example content",
                        "score": 0.75,
                    }
                ]
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(web_search_module, "AsyncTavilyClient", FakeClient)

    results = await web_search("example", top_k=1)

    assert results == [
        SearchResult(url="https://example.com", title="Example", content="Example content", relevance_score=0.75)
    ]


@pytest.mark.asyncio
async def test_web_search_tool_is_langchain_callable(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    results = await web_search_tool.ainvoke({"query": "example", "top_k": 1})

    assert results == []
