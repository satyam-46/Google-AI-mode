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
    sub_questions = _normalize_sub_questions([item.model_dump() for item in planner_output.sub_questions], query)
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
                details={"run_id": state.get("run_id"), "sub_questions": len(capped), "raw_sub_questions": len(sub_questions)},
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


def _normalize_sub_questions(sub_questions: list[dict[str, Any]], original_query: str) -> list[dict[str, Any]]:
    """Drop vague fragments that came from punctuation-only splitting."""
    normalized: list[dict[str, Any]] = []
    for sub_question in sub_questions:
        question = str(sub_question.get("question") or "")
        search_query = str(sub_question.get("search_query") or question)
        if normalized and _is_vague_direction_fragment(question, search_query):
            previous = normalized[-1]
            previous["question"] = _ensure_question(original_query)
            previous["search_query"] = original_query.strip(" ?.")
            previous["reasoning"] = (
                f"{previous.get('reasoning', '')} Folded a vague direction fragment into the main question."
            ).strip()
            continue
        normalized.append(sub_question)
    return normalized or [{"id": "q1", "question": _ensure_question(original_query), "search_query": original_query}]


def _is_vague_direction_fragment(question: str, search_query: str) -> bool:
    vague = {"which direction", "what direction", "which way", "what way", "direction"}
    candidates = {
        question.lower().strip(" ?."),
        search_query.lower().strip(" ?."),
        f"{question} {search_query}".lower().strip(" ?."),
    }
    return bool(candidates & vague)


def _ensure_question(text: str) -> str:
    stripped = text.strip()
    return stripped if stripped.endswith("?") else f"{stripped}?"


def _fanout_cap(complexity: float) -> int:
    if complexity < 0.3:
        return 2
    if complexity < 0.7:
        return 5
    return 8


def _now_ms() -> int:
    return int(time.time() * 1000)
