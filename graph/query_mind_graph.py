"""Phase 2 LangGraph assembly for QueryMind."""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from graph.checkpointer import get_checkpointer
from graph.nodes.arbitrator import arbitrator_node, route_after_arbitrator
from graph.nodes.cache import cache_lookup_node, cache_store_node, route_after_cache_lookup
from graph.nodes.critic import critic_node, route_after_critic
from graph.nodes.planner import planner_node
from graph.nodes.retriever import retriever_node, route_to_retrievers
from graph.nodes.session import initialize_node
from graph.nodes.synthesizer import synthesizer_node
from graph.state import QueryMindState


def build_graph(env: str = "dev", checkpointer: Any | None = None):
    graph = StateGraph(QueryMindState)

    graph.add_node("initialize", initialize_node)
    graph.add_node("cache_lookup", cache_lookup_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("arbitrator", arbitrator_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("cache_store", cache_store_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "cache_lookup")
    graph.add_conditional_edges(
        "cache_lookup",
        route_after_cache_lookup,
        {"cache_hit": "synthesizer", "cache_miss": "planner"},
    )
    graph.add_conditional_edges("planner", route_to_retrievers, ["retriever"])
    graph.add_edge("retriever", "arbitrator")
    graph.add_conditional_edges("arbitrator", route_after_arbitrator, {"synthesize": "synthesizer"})
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"approved": "cache_store", "human_review": END})
    graph.add_edge("cache_store", END)

    return graph.compile(checkpointer=checkpointer or get_checkpointer(env))


@lru_cache(maxsize=1)
def get_graph():
    return build_graph("dev")


def graph_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


async def run_query_graph(query: str, session_id: str | None = None, graph: Any | None = None) -> dict[str, Any]:
    session_id = session_id or str(uuid.uuid4())
    app = graph or get_graph()
    state = await app.ainvoke(
        {"original_query": query, "session_id": session_id},
        config=graph_config(session_id),
    )
    return project_current_run_state(dict(state))


async def stream_query_graph_events(
    query: str,
    session_id: str | None = None,
    graph: Any | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream graph events and answer tokens from the compiled LangGraph app."""
    session_id = session_id or str(uuid.uuid4())
    app = graph or get_graph()
    emitted_tokens = False

    async for event in app.astream_events(
        {"original_query": query, "session_id": session_id},
        config=graph_config(session_id),
        version="v2",
    ):
        name = str(event.get("name") or "")
        kind = str(event.get("event") or "")
        if kind in {"on_chain_start", "on_chain_end"} and name in {
            "initialize",
            "cache_lookup",
            "planner",
            "retriever",
            "arbitrator",
            "synthesizer",
            "critic",
            "cache_store",
        }:
            yield {"event": kind, "node": name, "session_id": session_id}

        if kind != "on_chain_stream" or name != "synthesizer":
            continue

        chunk = event.get("data", {}).get("chunk")
        if not isinstance(chunk, dict):
            continue
        answer_text = chunk.get("streaming_answer") or chunk.get("final_answer", {}).get("answer_text", "")
        for token in str(answer_text).split():
            emitted_tokens = True
            yield {"event": "token", "token": token, "session_id": session_id}

    if not emitted_tokens:
        state = project_current_run_state(dict(get_query_graph_state(session_id, app).values))
        answer_text = state.get("streaming_answer") or state.get("final_answer", {}).get("answer_text", "")
        for token in str(answer_text).split():
            yield {"event": "token", "token": token, "session_id": session_id}


def get_projected_query_graph_state(session_id: str, graph: Any | None = None) -> dict[str, Any]:
    checkpoint = get_query_graph_state(session_id, graph)
    return project_current_run_state(dict(checkpoint.values)) if checkpoint else {}


async def resume_query_graph(
    session_id: str,
    feedback: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    app = graph or get_graph()
    config = graph_config(session_id)
    resume_payload = feedback or {"approved": True}
    app.update_state(config, {"human_feedback": resume_payload})
    state = await app.ainvoke(Command(resume=resume_payload), config=config)
    return project_current_run_state(dict(state))


def get_query_graph_state(session_id: str, graph: Any | None = None) -> Any:
    app = graph or get_graph()
    return app.get_state(graph_config(session_id))


def project_current_run_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a clean current-run view while keeping session history."""
    interrupts = _serialize_interrupts(state.pop("__interrupt__", None))
    run_id = state.get("run_id")
    if not run_id:
        if interrupts:
            state["interrupts"] = interrupts
            state["requires_human_review"] = True
        return state

    projected = dict(state)
    for key in ("retrieval_results", "retriever_errors", "agent_traces"):
        items = state.get(key)
        if isinstance(items, list):
            projected[key] = _current_run_items(items, run_id)

    if interrupts:
        projected["interrupts"] = interrupts
        projected["requires_human_review"] = True

    return projected


def _current_run_items(items: list[Any], run_id: str) -> list[Any]:
    current = [
        item
        for item in items
        if isinstance(item, dict)
        and (item.get("run_id") == run_id or item.get("details", {}).get("run_id") == run_id)
    ]
    if current:
        return current
    return [
        item
        for item in items
        if not isinstance(item, dict)
        or (not item.get("run_id") and not item.get("details", {}).get("run_id"))
    ]


def _serialize_interrupts(interrupts: Any) -> list[dict[str, Any]]:
    if not interrupts:
        return []
    if not isinstance(interrupts, list):
        interrupts = [interrupts]

    serialized = []
    for interrupt in interrupts:
        serialized.append(
            {
                "id": str(getattr(interrupt, "id", "")),
                "value": getattr(interrupt, "value", interrupt),
            }
        )
    return serialized
