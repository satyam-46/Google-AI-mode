import asyncio

import pytest
from langgraph.types import Command

from core.chains.parsers import RetrievalEvidence
from graph.nodes.arbitrator import conflict_detector, score_conflict_sources
from graph.nodes.cache import get_session_store
from graph.nodes.planner import planner_node
from graph.nodes.retriever import retriever_node
from graph.nodes.synthesizer import synthesizer_node
from graph.query_mind_graph import build_graph, graph_config, resume_query_graph, run_query_graph, stream_query_graph_events


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
async def test_run_query_graph_serializes_interrupts_for_api_state():
    graph = build_graph("test")
    state = await run_query_graph(
        "What is an unknown local-only interrupt topic?",
        session_id="phase2-serialized-interrupt",
        graph=graph,
    )

    assert "__interrupt__" not in state
    assert state["requires_human_review"] is True
    assert state["interrupts"]
    assert state["interrupts"][0]["value"]["reason"] == "low_confidence_or_sensitive_topic"


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


@pytest.mark.asyncio
async def test_planner_merges_vague_direction_fragment():
    result = await planner_node({"original_query": "where does sun rise from, which direction?"})

    assert len(result["sub_questions"]) == 1
    assert result["sub_questions"][0]["question"] == "where does sun rise from, which direction?"


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


def test_source_aware_arbitration_scores_authority_recency_and_corroboration():
    scores = score_conflict_sources(
        {
            "claim_a": "Python 2 was released in 2000.",
            "source_a": "https://www.python.org/download/releases/2.0/",
            "claim_b": "Python 2 was released in 1991.",
            "source_b": "https://www.quora.com/python-release-date",
            "corroborating_sources": ["The Python 2.0 release happened in 2000."],
        }
    )

    assert scores["claim_a"]["authority"] > scores["claim_b"]["authority"]
    assert scores["claim_a"]["corroboration"] > scores["claim_b"]["corroboration"]
    assert set(scores["claim_a"]) == {"authority", "recency", "corroboration", "total"}


@pytest.mark.asyncio
async def test_arbitrator_node_uses_arbitrator_chain(monkeypatch):
    import graph.nodes.arbitrator as arbitrator_module

    class FakeArbitratorChain:
        async def ainvoke(self, payload):
            assert payload["conflict"]["entity"] == "Python"

            class Result:
                def model_dump(self):
                    return {
                        "entity": "Python",
                        "winning_claim": "Python 2 was released in 2000.",
                        "winning_source": "https://example.com/a",
                        "reasoning": "Source A is more authoritative.",
                        "confidence": 0.9,
                    }

            return Result()

    monkeypatch.setattr(arbitrator_module, "arbitrator_chain", FakeArbitratorChain())
    result = await arbitrator_module.arbitrator_node(
        {
            "retrieval_results": [
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
        }
    )

    assert result["arbitration_results"][0]["reasoning"].startswith("Source A is more authoritative.")
    assert "source_scores" in result["arbitration_results"][0]


@pytest.mark.asyncio
async def test_graph_event_stream_yields_node_events_and_tokens():
    graph = build_graph("test")
    events = [
        event
        async for event in stream_query_graph_events(
            query="What is an unknown local-only streaming topic?",
            session_id="phase2-stream-events",
            graph=graph,
        )
    ]

    assert any(event["event"] == "on_chain_start" and event["node"] == "planner" for event in events)
    assert any(event["event"] == "token" for event in events)


@pytest.mark.asyncio
async def test_synthesizer_marks_partial_evidence_low_confidence():
    result = await synthesizer_node(
        {
            "original_query": "Where does the sun set?",
            "run_id": "run-partial",
            "retrieval_results": [
                {
                    "run_id": "run-partial",
                    "sub_question_id": "q1",
                    "answer_text": "The sun generally sets in the west.",
                    "citations": [],
                    "confidence": 0.55,
                    "status": "partial",
                }
            ],
        }
    )

    assert result["final_answer"]["answer_text"].startswith("[LOW CONFIDENCE]")


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
async def test_same_session_second_query_projects_only_current_run_results(monkeypatch):
    import graph.nodes.retriever as retriever_module

    class RunAwareRetrieverChain:
        async def ainvoke(self, payload):
            return RetrievalEvidence(
                sub_question_id=payload["sub_question_id"],
                answer_text=f"Evidence for {payload['sub_question_id']}",
                citations=[],
                confidence=0.8,
            )

    monkeypatch.setattr(retriever_module, "retriever_chain", RunAwareRetrieverChain())
    graph = build_graph("test")
    session_id = "same-session"

    first = await run_query_graph("Compare Alpha and Beta", session_id=session_id, graph=graph)
    second = await run_query_graph("Where does the sun rise?", session_id=session_id, graph=graph)

    assert first["run_id"] != second["run_id"]
    assert first["retrieval_results"]
    assert second["retrieval_results"]
    assert {item["run_id"] for item in second["retrieval_results"]} == {second["run_id"]}
    assert all(trace["details"].get("run_id") == second["run_id"] for trace in second["agent_traces"])
    assert second["session_history"]
    assert second["session_history"][-1]["run_id"] == first["run_id"]


@pytest.mark.asyncio
async def test_session_cache_uses_embeddings_when_available(tmp_path, monkeypatch):
    from core.memory.session_store import SessionStore

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    class FakeEmbeddings:
        async def aembed_query(self, query):
            return [1.0, 0.0] if "alpha" in query else [0.0, 1.0]

    monkeypatch.setattr(SessionStore, "_create_embeddings", staticmethod(lambda: FakeEmbeddings()))
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    await store.store_result(
        "session",
        "alpha topic",
        {"final_answer": {"answer_text": "alpha answer"}, "retrieval_results": []},
    )

    cached = await store.get_cached("session", "alpha related", threshold=0.9)

    assert cached is not None
    assert cached["cache_hit"]["method"] == "embedding"


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
    assert result["retrieval_results"][0]["error"] == "extraction timed out after 0.001s"
    assert result["retriever_errors"]
    assert result["retriever_errors"][0]["stage"] == "extraction"
    assert result["retriever_errors"][0]["type"] == "TimeoutError"
    assert result["retriever_errors"][0]["elapsed_ms"] >= 0
    assert result["retriever_errors"][0]["sub_question_id"] == "q-timeout"


@pytest.mark.asyncio
async def test_retriever_timeout_preserves_search_evidence(monkeypatch):
    import graph.nodes.retriever as retriever_module

    class HangingRetrieverChain:
        async def ainvoke(self, payload):
            await asyncio.sleep(1)

    async def fake_web_search(query, top_k=3):
        return [
            retriever_module.SearchResult(
                url="https://example.com/sunrise",
                title="Sunrise",
                content="The sun generally rises in the east.",
                relevance_score=0.9,
            )
        ]

    monkeypatch.setattr(retriever_module, "retriever_chain", HangingRetrieverChain())
    monkeypatch.setattr(retriever_module, "web_search", fake_web_search)

    result = await retriever_node(
        {
            "sub_question": {"id": "q-sunrise", "question": "Where does the sun rise?"},
            "retriever_timeout": 0.001,
        }
    )

    retrieval = result["retrieval_results"][0]
    assert retrieval["status"] == "partial"
    assert "east" in retrieval["answer_text"]
    assert len(retrieval["answer_text"]) < 300
    assert retrieval["citations"]
    assert retrieval["error_details"]["stage"] == "extraction"
    assert retrieval["error_details"]["search_results"] == 1


@pytest.mark.asyncio
async def test_retriever_surfaces_search_stage_errors(monkeypatch):
    import graph.nodes.retriever as retriever_module

    async def failing_web_search(query, top_k=3):
        raise RuntimeError("tavily exploded")

    monkeypatch.setattr(retriever_module, "web_search", failing_web_search)

    result = await retriever_node(
        {
            "sub_question": {"id": "q-search", "question": "Will search fail?"},
            "run_id": "run-search-error",
        }
    )

    retrieval = result["retrieval_results"][0]
    error = result["retriever_errors"][0]
    assert retrieval["status"] == "failed"
    assert error["stage"] == "search"
    assert error["type"] == "RuntimeError"
    assert error["raw_message"] == "tavily exploded"
    assert "RuntimeError: tavily exploded" in error["traceback_tail"]
