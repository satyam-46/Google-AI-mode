"""Phase 3 ADK orchestration layer for QueryMind."""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from adk.callbacks import (
    after_model_callback,
    before_model_callback,
    on_model_error_callback,
    record_event,
)
from adk.grounding import GroundedAnswer, GroundingProvider, answer_with_grounding
from adk.router import QueryRoute, RouteDecision, route_query_text
from adk.tools.querymind_tool import querymind_tool, run_querymind

DeepProvider = Callable[[str, str], Awaitable[dict[str, Any]]]


async def route_query(
    query: str,
    session_id: str | None = None,
    *,
    has_session_context: bool = False,
    session_context: dict[str, Any] | None = None,
    fast_provider: GroundingProvider | None = None,
    deep_provider: DeepProvider | None = None,
) -> dict[str, Any]:
    """Route one user turn to fast grounding or the full QueryMind graph."""
    session_id = session_id or str(uuid.uuid4())
    routed_query = resolve_follow_up_query(query, session_context or {})
    decision = route_query_text(routed_query, has_session_context=has_session_context or bool(session_context))
    record_event(
        "route_decision",
        session_id=session_id,
        route=decision.route.value,
        original_query=query,
        routed_query=routed_query,
        complexity_score=decision.complexity_score,
        matched_rules=decision.matched_rules,
    )

    if decision.route is QueryRoute.CLARIFY:
        return _clarify_response(session_id, decision)

    if decision.route is QueryRoute.FAST:
        provider = fast_provider or answer_with_grounding
        try:
            answer = await provider(routed_query, session_id)
        except Exception as exc:
            return _error_response(session_id, decision, "fast_grounding", exc)
        record_event(
            "route_complete",
            session_id=session_id,
            route=decision.route.value,
            confidence=answer.confidence,
            grounding_chunks=len(answer.citations),
        )
        return _fast_response(session_id, decision, answer)

    provider = deep_provider or run_querymind
    try:
        result = await provider(routed_query, session_id)
    except Exception as exc:
        return _error_response(session_id, decision, "querymind", exc)
    confidence = float(result.get("confidence", 0.0))
    record_event("route_complete", session_id=session_id, route=decision.route.value, confidence=confidence)
    return _deep_response(session_id, decision, result)


def build_root_agent(model: str = "gemini-1.5-flash") -> Any:
    """Build the ADK root agent with QueryMind exposed as a tool."""
    from google.adk.agents import LlmAgent
    from google.adk.tools import google_search

    tools = [google_search]
    if querymind_tool is not None:
        tools.append(querymind_tool)

    return LlmAgent(
        name="querymind_orchestrator",
        model=model,
        description="Routes simple factual lookup and complex research queries.",
        instruction=(
            "Use Google Search grounding for simple factual lookup. Use QueryMind "
            "for complex, comparative, sensitive, conflicting, or multi-source research. "
            "Always preserve citations and confidence."
        ),
        tools=tools,
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        on_model_error_callback=on_model_error_callback,
    )


def _clarify_response(session_id: str, decision: RouteDecision) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "route": decision.route.value,
        "answer": "Can you clarify what this refers to?",
        "citations": [],
        "confidence": 0.0,
        "requires_human_review": False,
        "route_decision": decision.model_dump(),
    }


def _error_response(session_id: str, decision: RouteDecision, stage: str, error: Exception) -> dict[str, Any]:
    structured_error = {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
    }
    record_event(
        "route_error",
        session_id=session_id,
        route=decision.route.value,
        **structured_error,
    )
    return {
        "session_id": session_id,
        "route": decision.route.value,
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "requires_human_review": True,
        "route_decision": decision.model_dump(),
        "error": structured_error,
    }


def _fast_response(session_id: str, decision: RouteDecision, answer: GroundedAnswer) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "route": decision.route.value,
        "answer": answer.answer,
        "citations": answer.citations,
        "confidence": answer.confidence,
        "requires_human_review": answer.confidence < 0.5,
        "grounding_metadata": answer.metadata,
        "route_decision": decision.model_dump(),
    }


def _deep_response(session_id: str, decision: RouteDecision, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "route": decision.route.value,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", 0.0),
        "requires_human_review": bool(result.get("requires_human_review", False)),
        "state": result.get("state", {}),
        "route_decision": decision.model_dump(),
    }


def resolve_follow_up_query(query: str, session_context: dict[str, Any]) -> str:
    """Rewrite small pronoun follow-ups with the last known topic when possible."""
    normalized = " ".join(query.strip().split())
    if not session_context or not normalized:
        return normalized
    lowered = normalized.lower()
    if " it" not in f" {lowered}" and not lowered.startswith(("it ", "who built it", "where does it")):
        return normalized

    topic = _last_topic(session_context)
    if not topic:
        return normalized
    return normalized.replace(" it", f" {topic}").replace("It ", f"{topic} ", 1)


def _last_topic(session_context: dict[str, Any]) -> str:
    for item in reversed(session_context.get("session_history") or []):
        query = str(item.get("query", ""))
        topic = _topic_from_query(query)
        if topic:
            return topic
    return _topic_from_query(str(session_context.get("original_query", "")))


def _topic_from_query(query: str) -> str:
    import re

    match = re.search(r"\b(?:about|is|are|was|were|built)\s+([A-Z][A-Za-z0-9_.-]+)", query)
    if match:
        return match.group(1)
    capitals = re.findall(r"\b[A-Z][A-Za-z0-9_.-]+\b", query)
    return capitals[-1] if capitals else ""


__all__ = ["build_root_agent", "resolve_follow_up_query", "route_query"]
