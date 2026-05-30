"""Phase 2 LangGraph assembly for QueryMind."""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

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
    return dict(state)


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
    return dict(state)


def get_query_graph_state(session_id: str, graph: Any | None = None) -> Any:
    app = graph or get_graph()
    return app.get_state(graph_config(session_id))
