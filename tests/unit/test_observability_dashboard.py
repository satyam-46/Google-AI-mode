from __future__ import annotations

import pytest

from core.memory.session_store import SessionStore
from observability.dashboard import (
    build_dashboard_model,
    build_session_trends,
    build_timeline,
    compare_replay,
    summarize_session,
)
from observability.tracer import TraceStore


def test_summarize_session_uses_answer_quality_and_trace_totals():
    summary = summarize_session(
        {
            "session_id": "session-a",
            "updated_at": 1_700_000_000,
            "state": {
                "original_query": "Compare Alpha and Beta",
                "final_answer": {
                    "answer_text": "Alpha and Beta differ.",
                    "confidence": 0.82,
                    "citations": [{"url": "https://example.com"}],
                },
                "requires_human_review": True,
                "sub_questions": [{"id": "q1"}, {"id": "q2"}],
                "retrieval_results": [{"id": "q1"}],
                "conflicts_detected": [{"entity": "Alpha"}],
                "cache_hits": ["previous"],
                "agent_traces": [
                    {"name": "planner", "start_ms": 100, "end_ms": 150, "tokens_used": 10},
                    {"name": "synthesizer", "start_ms": 160, "end_ms": 240, "tokens_used": 20},
                ],
            },
        }
    )

    assert summary["query"] == "Compare Alpha and Beta"
    assert summary["confidence"] == 0.82
    assert summary["requires_human_review"] is True
    assert summary["citations"] == 1
    assert summary["sub_questions"] == 2
    assert summary["retrievals"] == 1
    assert summary["conflicts"] == 1
    assert summary["trace_steps"] == 2
    assert summary["latency_ms"] == 140
    assert summary["tokens"] == 30
    assert summary["estimated_cost_usd"] > 0
    assert summary["cache_hits"] == 1


def test_build_dashboard_model_aggregates_sessions_and_events():
    model = build_dashboard_model(
        [
            {
                "session_id": "high",
                "updated_at": 20,
                "state": {"final_answer": {"confidence": 0.9, "citations": [{"url": "a"}]}},
            },
            {
                "session_id": "review",
                "updated_at": 10,
                "state": {"confidence_score": {"score": 0.3}, "requires_human_review": True},
            },
        ],
        events=[
            {"event": "old", "timestamp_ms": 1},
            {"event": "new", "timestamp_ms": 2},
        ],
    )

    assert model["metrics"]["total_sessions"] == 2
    assert model["metrics"]["average_confidence"] == 0.6
    assert model["metrics"]["human_review"] == 1
    assert model["metrics"]["citations"] == 1
    assert model["metrics"]["estimated_cost_usd"] == 0
    assert [event["event"] for event in model["events"]] == ["new", "old"]
    assert "high" in model["session_records"]


def test_dashboard_builds_timeline_and_session_trends():
    timeline = build_timeline(
        [
            {"name": "planner", "start_ms": 100, "end_ms": 150, "tokens_used": 10},
            {
                "name": "retriever",
                "start_ms": 120,
                "end_ms": 180,
                "details": {"status": "partial"},
            },
        ]
    )
    trends = build_session_trends(
        [
            {"updated": "later", "updated_at": 2, "confidence": 0.8, "latency_ms": 20, "cache_hit_rate": 0.5, "requires_human_review": False, "estimated_cost_usd": 0.001},
            {"updated": "earlier", "updated_at": 1, "confidence": 0.6, "latency_ms": 30, "cache_hit_rate": 0.0, "requires_human_review": True, "estimated_cost_usd": 0.002},
        ]
    )

    assert timeline[0]["agent"] == "planner"
    assert timeline[0]["offset_ms"] == 0
    assert timeline[0]["duration_ms"] == 50
    assert timeline[1]["status"] == "partial"
    assert [row["updated"] for row in trends] == ["earlier", "later"]


def test_compare_replay_reports_quality_deltas():
    comparison = compare_replay(
        {"confidence": 0.7, "citations": 1, "answer": "saved"},
        {
            "route": "deep",
            "answer": "replayed answer",
            "confidence": 0.9,
            "citations": [{"url": "a"}, {"url": "b"}],
            "requires_human_review": False,
        },
    )

    assert comparison["confidence_delta"] == 0.2
    assert comparison["citation_delta"] == 1
    assert comparison["route"] == "deep"
    assert comparison["replay_answer_chars"] > comparison["original_answer_chars"]


@pytest.mark.asyncio
async def test_session_store_lists_recent_sessions(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")

    await store.save("session-a", {"original_query": "A"})
    await store.save("session-b", {"original_query": "B"})

    sessions = await store.list_sessions(limit=10)

    assert {session["session_id"] for session in sessions} == {"session-a", "session-b"}
    assert {session["state"]["original_query"] for session in sessions} == {"A", "B"}
    assert all("updated_at" in session for session in sessions)


@pytest.mark.asyncio
async def test_trace_store_records_graph_state(tmp_path):
    store = TraceStore(db_path=tmp_path / "traces.sqlite3")

    await store.record_graph_state(
        {
            "session_id": "session-a",
            "run_id": "run-a",
            "agent_traces": [
                {"name": "planner", "start_ms": 100, "end_ms": 140, "tokens_used": 12},
                {"name": "critic", "start_ms": 150, "end_ms": 170, "tokens_used": 3},
            ],
        }
    )

    records = await store.list_records(limit=10)

    assert [record["node"] for record in records] == ["critic", "planner"]
    assert {record["session_id"] for record in records} == {"session-a"}
    assert {record["run_id"] for record in records} == {"run-a"}
    assert sum(record["tokens_used"] for record in records) == 15
