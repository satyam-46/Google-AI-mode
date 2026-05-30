import asyncio

import pytest
from langgraph.types import Command

from core.chains.parsers import RetrievalEvidence
from graph.nodes.arbitrator import conflict_detector
from graph.nodes.cache import get_session_store
from graph.nodes.retriever import retriever_node
from graph.query_mind_graph import build_graph, graph_config, resume_query_graph


@pytest.fixture(autouse=True)
def isolated_session_db(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYMIND_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    get_session_store.cache_clear()
    yield
    get_session_store.cache_clear()


@pytest.mark.asyncio
async def test_phase2_graph_executes_end_to_end_after_review_resume():
    graph = build_graph("test")
    session_id = "phase2-end-to-end"

    interrupted = await graph.ainvoke(
        {"original_query": "What is an unknown local-only test topic?", "session_id": session_id},
        config=graph_config(session_id),
    )

    assert "__interrupt__" in interrupted
    assert graph.get_state(graph_config(session_id)).next == ("critic",)

    resumed = await resume_query_graph(session_id=session_id, feedback={"approved": True}, graph=graph)

    assert resumed["final_answer"]["answer_text"].startswith("[LOW CONFIDENCE]")
    assert resumed["requires_human_review"] is False
    assert resumed["human_feedback"] == {"approved": True}
    assert graph.get_state(graph_config(session_id)).next == ()
    assert {"planner", "retriever", "arbitrator", "synthesizer", "critic", "cache_store"} <= {
        trace["name"] for trace in resumed["agent_traces"]
    }


@pytest.mark.asyncio
async def test_parallel_retrievers_have_overlapping_trace_windows(monkeypatch):
    import graph.nodes.retriever as retriever_module

    class SlowRetrieverChain:
        async def ainvoke(self, payload):
            await asyncio.sleep(0.05)
            return RetrievalEvidence(
                sub_question_id=payload["sub_question_id"],
                answer_text=f"Evidence for {payload['question']}",
                citations=[],
                confidence=0.8,
            )

    monkeypatch.setattr(retriever_module, "retriever_chain", SlowRetrieverChain())
    graph = build_graph("test")
    session_id = "phase2-parallel"
    state = await graph.ainvoke(
        {"original_query": "Compare Paris and Berlin", "session_id": session_id},
        config=graph_config(session_id),
    )

    retriever_traces = [trace for trace in state["agent_traces"] if trace["name"] == "retriever"]
    assert len(retriever_traces) >= 2
    latest_start = max(trace["start_ms"] for trace in retriever_traces)
    earliest_end = min(trace["end_ms"] for trace in retriever_traces)
    assert latest_start < earliest_end


def test_conflict_detection_triggers_on_contradictory_dates():
    conflicts = conflict_detector(
        [
            {
                "sub_question_id": "q1",
                "answer_text": "Python 2 was released in 2000.",
                "citations": [{"source": "A", "url": "https://example.com/a"}],
                "confidence": 0.9,
            },
            {
                "sub_question_id": "q2",
                "answer_text": "Python 2 was released in 1991.",
                "citations": [{"source": "B", "url": "https://example.com/b"}],
                "confidence": 0.9,
            },
        ]
    )

    assert conflicts
    assert conflicts[0]["entity"] == "Python"


@pytest.mark.asyncio
async def test_cache_hit_skips_planner_after_first_completed_run():
    graph = build_graph("test")
    session_id = "phase2-cache"
    query = "What is a cacheable local-only test topic?"
    await get_session_store().store_result(
        session_id=session_id,
        query=query,
        result={
            "retrieval_results": [
                {
                    "sub_question_id": "q-cache",
                    "answer_text": "Cached evidence is available.",
                    "citations": [],
                    "confidence": 0.8,
                    "status": "ok",
                }
            ],
            "final_answer": {"answer_text": "Cached answer.", "citations": [], "confidence": 0.8},
        },
    )

    cached = await graph.ainvoke({"original_query": query, "session_id": session_id}, config=graph_config(session_id))

    assert cached["cache_hits"]
    trace_names = [trace["name"] for trace in cached["agent_traces"]]
    assert "cache_lookup" in trace_names
    assert "planner" not in trace_names


@pytest.mark.asyncio
async def test_retriever_timeout_returns_failed_result(monkeypatch):
    import graph.nodes.retriever as retriever_module

    class HangingRetrieverChain:
        async def ainvoke(self, payload):
            await asyncio.sleep(1)

    monkeypatch.setattr(retriever_module, "retriever_chain", HangingRetrieverChain())

    result = await retriever_node(
        {
            "sub_question": {"id": "q-timeout", "question": "Will this timeout?"},
            "retriever_timeout": 0.001,
        }
    )

    assert result["retrieval_results"][0]["status"] == "failed"
    assert result["retriever_errors"]
