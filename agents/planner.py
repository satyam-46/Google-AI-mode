"""Planner agent node."""

from __future__ import annotations

from typing import Any

from core.chains.base_chains import planner_chain


async def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Decompose the original query into typed sub-questions."""
    query = state.get("original_query", "")
    planner_output = await planner_chain.ainvoke({"query": query})
    sub_questions = [item.model_dump() for item in planner_output.sub_questions]

    return {
        "sub_questions": sub_questions,
        "complexity_score": min(len(sub_questions) / 5, 1.0),
    }
