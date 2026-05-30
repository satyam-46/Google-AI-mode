"""Session cache nodes."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from core.memory.session_store import SessionStore
from graph.state import AgentTrace, QueryMindState


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    return SessionStore()


async def cache_lookup_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    session_id = state.get("session_id", "")
    query = state.get("original_query", "")
    cached = await get_session_store().get_cached(session_id=session_id, query=query)
    if not cached:
        return {
            "cache_hits": [],
            "agent_traces": [
                AgentTrace(name="cache_lookup", start_ms=start, end_ms=_now_ms(), details={"hit": False}).model_dump()
            ],
        }

    hit = cached["cache_hit"]
    return {
        "retrieval_results": cached.get("retrieval_results", []),
        "final_answer": cached.get("final_answer", {}),
        "cache_hits": [hit["query_hash"]],
        "agent_traces": [
            AgentTrace(
                name="cache_lookup",
                start_ms=start,
                end_ms=_now_ms(),
                details={"hit": True, "similarity": hit["similarity"]},
            ).model_dump()
        ],
    }


async def cache_store_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    session_id = state.get("session_id", "")
    query = state.get("original_query", "")
    result = {
        "retrieval_results": state.get("retrieval_results", []),
        "final_answer": state.get("final_answer", {}),
    }
    if session_id and query and state.get("final_answer"):
        await get_session_store().store_result(session_id=session_id, query=query, result=result)
        await get_session_store().save(session_id=session_id, state=dict(state))

    return {
        "total_latency_ms": max(0, _now_ms() - int(state.get("started_at_ms", start))),
        "agent_traces": [
            AgentTrace(name="cache_store", start_ms=start, end_ms=_now_ms(), details={"stored": bool(result)}).model_dump()
        ],
    }


def route_after_cache_lookup(state: QueryMindState) -> str:
    return "cache_hit" if state.get("cache_hits") else "cache_miss"


def _now_ms() -> int:
    return int(time.time() * 1000)
