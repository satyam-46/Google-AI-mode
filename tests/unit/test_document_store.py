import pytest

from core.tools.document_store import Document, DocumentStore


@pytest.mark.asyncio
async def test_add_and_search_ranks_relevant_document_first(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    store = DocumentStore()
    docs = [
        Document(id="d1", text="Paris is the capital of France."),
        Document(id="d2", text="Berlin is the capital of Germany."),
    ]

    await store.add_documents(docs)
    results = await store.semantic_search("capital France", top_k=2)

    assert results
    assert "Paris" in results[0].text


@pytest.mark.asyncio
async def test_semantic_search_tool_returns_text(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    store = DocumentStore()
    await store.add_documents([Document(id="d1", text="LangChain uses LCEL runnables.")])

    tool = store.as_tool()
    results = await tool.ainvoke({"query": "LCEL", "top_k": 1})

    assert results == ["LangChain uses LCEL runnables."]
