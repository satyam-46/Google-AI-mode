import os
import time
import uuid

import pytest
from dotenv import load_dotenv

load_dotenv(".env")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LANGSMITH_TESTS") != "true" or not os.getenv("LANGSMITH_API_KEY"),
    reason="Set RUN_LANGSMITH_TESTS=true and LANGSMITH_API_KEY to verify live LangSmith traces.",
)


@pytest.mark.asyncio
async def test_langsmith_records_phase2_graph_topology(monkeypatch):
    from langsmith import Client

    from graph.query_mind_graph import build_graph, graph_config

    monkeypatch.setenv("QUERYMIND_FORCE_FAKE_LLM", "true")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    marker = f"phase2-langsmith-test-{uuid.uuid4()}"
    graph = build_graph("test")
    config = graph_config(marker)
    config["tags"] = ["phase2-langsmith-integration", marker]
    config["metadata"] = {"marker": marker}
    await graph.ainvoke({"original_query": "LangSmith integration topology test", "session_id": marker}, config=config)

    client = Client()
    project = os.getenv("LANGSMITH_PROJECT") or "default"
    runs = []
    for _ in range(12):
        runs = list(client.list_runs(project_name=project, filter=f'has(tags, "{marker}")', limit=100))
        if any(run.name == "LangGraph" for run in runs):
            break
        time.sleep(2)

    names = {run.name for run in runs}
    assert "LangGraph" in names
    assert {"initialize", "cache_lookup", "planner", "retriever", "arbitrator", "synthesizer", "critic"} <= names
    assert all(run.start_time for run in runs)
