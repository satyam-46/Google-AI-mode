import pytest


from core.tools.web_search import web_search, SearchResult


@pytest.mark.asyncio
async def test_web_search_returns_list():
    results = await web_search("test query", top_k=2)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_searchresult_model_empty_ok():
    # Ensure SearchResult model is usable even when no results are returned
    results = await web_search("another test", top_k=1)
    for r in results:
        assert isinstance(r, SearchResult)
