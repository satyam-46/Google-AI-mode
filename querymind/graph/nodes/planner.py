"""Planner node for adaptive query fan-out."""

from __future__ import annotations

import re
import time
from typing import Any

from core.chains.base_chains import planner_chain
from graph.state import AgentTrace, QueryMindState

_COMPARISON_WORDS = {"compare", "versus", "vs", "difference", "better", "best", "between"}
_TEMPORAL_WORDS = {"latest", "recent", "today", "yesterday", "history", "timeline", "release", "date"}
_CAUSAL_WORDS = {"why", "how", "because", "impact", "effect", "reason", "cause"}


async def planner_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    query = state.get("original_query", "")
    planner_output = await planner_chain.ainvoke({"query": query})
    sub_questions = [item.model_dump() for item in planner_output.sub_questions]
    complexity = compute_complexity(query, sub_questions)
    capped = sub_questions[:_fanout_cap(complexity)]

    return {
        "sub_questions": capped,
        "complexity_score": complexity,
        "agent_traces": [
            AgentTrace(
                name="planner",
                start_ms=start,
                end_ms=_now_ms(),
                details={"sub_questions": len(capped), "raw_sub_questions": len(sub_questions)},
            ).model_dump()
        ],
    }


def compute_complexity(query: str, sub_questions: list[dict[str, Any]]) -> float:
    words = {word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", query)}
    title_entities = len(re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", query))
    score = 0.12 * max(len(sub_questions), 1)
    score += 0.08 * min(title_entities, 4)
    score += 0.18 if words & _COMPARISON_WORDS else 0.0
    score += 0.16 if words & _TEMPORAL_WORDS else 0.0
    score += 0.14 if words & _CAUSAL_WORDS else 0.0
    score += 0.12 if len(query) > 120 else 0.0
    return min(score, 1.0)


def _fanout_cap(complexity: float) -> int:
    if complexity < 0.3:
        return 2
    if complexity < 0.7:
        return 5
    return 8


def _now_ms() -> int:
    return int(time.time() * 1000)
