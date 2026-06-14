from __future__ import annotations

from typing import Any

import pytest

from adk.callbacks import clear_events, get_events
from adk.grounding import GroundedAnswer, _grounding_citations
from adk.orchestrator import build_root_agent, resolve_follow_up_query, route_query
from adk.router import QueryRoute, route_query_text
from adk.session_manager import LocalSessionManager


@pytest.mark.parametrize(
    ("query", "expected", "rule"),
    [
        ("Where does the sun rise?", QueryRoute.FAST, "direct_where"),
        ("When was Python 3 released?", QueryRoute.FAST, "direct_when"),
        ("Who built LangGraph?", QueryRoute.FAST, "direct_who"),
        ("What is LangChain?", QueryRoute.FAST, "direct_what"),
        ("Define vector database", QueryRoute.FAST, "definition"),
        ("What is the capital of France?", QueryRoute.FAST, "capital"),
        ("Compare Python 2 vs Python 3 release dates", QueryRoute.DEEP, "comparison"),
        ("Research the evidence for RAG vs long context and cite sources", QueryRoute.DEEP, "research"),
        ("What is the latest OpenAI model pricing today?", QueryRoute.DEEP, "current"),
        ("Should I invest in Tesla stock?", QueryRoute.DEEP, "sensitive"),
    ],
)
def test_rule_router_covers_simple_and_complex_queries(query: str, expected: QueryRoute, rule: str):
    decision = route_query_text(query)

    assert decision.route is expected
    assert rule in decision.matched_rules
    assert 0 <= decision.complexity_score <= 1


def test_follow_up_without_context_clarifies():
    decision = route_query_text("Who built it?")

    assert decision.route is QueryRoute.CLARIFY
    assert "follow_up" in decision.matched_rules


def test_follow_up_with_context_can_use_fast_path():
    decision = route_query_text("Who built it?", has_session_context=True)

    assert decision.route is QueryRoute.FAST
    assert "follow_up" in decision.matched_rules


@pytest.mark.asyncio
async def test_orchestrator_fast_path_skips_deep_provider():
    clear_events()
    called = {"deep": False}

    async def fast_provider(query: str, session_id: str) -> GroundedAnswer:
        return GroundedAnswer(
            answer="The Sun generally rises in the east.",
            citations=[{"source": "test", "url": "", "excerpt": "east"}],
            confidence=0.9,
            metadata={"mode": "test"},
        )

    async def deep_provider(query: str, session_id: str) -> dict[str, Any]:
        called["deep"] = True
        return {}

    result = await route_query(
        "Where does the sun rise?",
        "phase3-fast",
        fast_provider=fast_provider,
        deep_provider=deep_provider,
    )

    assert result["route"] == "fast"
    assert result["answer"] == "The Sun generally rises in the east."
    assert result["citations"]
    assert called["deep"] is False
    assert any(event["event"] == "route_decision" for event in get_events())


@pytest.mark.asyncio
async def test_orchestrator_deep_path_uses_querymind_provider():
    called = {"deep": False}

    async def deep_provider(query: str, session_id: str) -> dict[str, Any]:
        called["deep"] = True
        return {
            "answer": "Deep answer",
            "citations": [{"source": "QueryMind", "url": "", "excerpt": "evidence"}],
            "confidence": 0.77,
            "state": {"session_id": session_id},
        }

    result = await route_query(
        "Compare Python 2 vs Python 3 release dates",
        "phase3-deep",
        deep_provider=deep_provider,
    )

    assert called["deep"] is True
    assert result["route"] == "deep"
    assert result["answer"] == "Deep answer"
    assert result["state"]["session_id"] == "phase3-deep"


@pytest.mark.asyncio
async def test_orchestrator_surfaces_structured_provider_errors():
    async def failing_fast_provider(query: str, session_id: str) -> GroundedAnswer:
        raise RuntimeError("grounding boom")

    result = await route_query(
        "Where does the sun rise?",
        "phase3-error",
        fast_provider=failing_fast_provider,
    )

    assert result["requires_human_review"] is True
    assert result["error"] == {
        "stage": "fast_grounding",
        "type": "RuntimeError",
        "message": "grounding boom",
    }


def test_follow_up_query_rewrites_from_session_context():
    rewritten = resolve_follow_up_query("Who built it?", {"original_query": "Tell me about LangGraph"})

    assert rewritten == "Who built LangGraph?"


def test_grounding_metadata_chunks_convert_to_citations():
    citations = _grounding_citations(
        {
            "grounding_chunks": [
                {"web": {"title": "Example Source", "uri": "https://example.com/source"}},
            ]
        }
    )

    assert citations == [{"source": "Example Source", "url": "https://example.com/source", "excerpt": ""}]


@pytest.mark.asyncio
async def test_local_session_manager_tracks_context():
    manager = LocalSessionManager()

    assert await manager.has_context("u1", "s1") is False
    session = await manager.update_state("u1", "s1", {"topic": "LangGraph"})

    assert session.state["topic"] == "LangGraph"
    assert await manager.has_context("u1", "s1") is True


def test_build_root_agent_wires_adk_agent():
    agent = build_root_agent()

    assert agent.name == "querymind_orchestrator"
    assert len(agent.tools) >= 1
