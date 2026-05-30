"""Conflict detection and arbitration node."""

from __future__ import annotations

import itertools
import re
import time
from typing import Any

from graph.state import AgentTrace, QueryMindState

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


async def arbitrator_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    results = state.get("retrieval_results", [])
    conflicts = conflict_detector(results)
    arbitration_results = [arbitrate_conflict(conflict) for conflict in conflicts]

    return {
        "conflicts_detected": conflicts,
        "arbitration_results": arbitration_results,
        "agent_traces": [
            AgentTrace(
                name="arbitrator",
                start_ms=start,
                end_ms=_now_ms(),
                details={"conflicts": len(conflicts)},
            ).model_dump()
        ],
    }


def conflict_detector(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for left, right in itertools.combinations(results, 2):
        left_text = str(left.get("answer_text", ""))
        right_text = str(right.get("answer_text", ""))
        if not left_text or not right_text:
            continue

        left_years = set(_YEAR_RE.findall(left_text))
        right_years = set(_YEAR_RE.findall(right_text))
        left_numbers = set(_NUMBER_RE.findall(left_text))
        right_numbers = set(_NUMBER_RE.findall(right_text))

        has_year_conflict = bool(left_years and right_years and left_years != right_years)
        has_number_conflict = bool(left_numbers and right_numbers and left_numbers != right_numbers)
        has_boolean_conflict = _boolean_conflict(left_text, right_text)
        if not (has_year_conflict or has_number_conflict or has_boolean_conflict):
            continue

        conflicts.append(
            {
                "entity": _shared_entity(left_text, right_text) or "unknown",
                "claim_a": left_text[:500],
                "source_a": _first_source(left),
                "claim_b": right_text[:500],
                "source_b": _first_source(right),
                "reason": "contradictory_year_or_value_or_boolean",
            }
        )
    return conflicts


def arbitrate_conflict(conflict: dict[str, Any]) -> dict[str, Any]:
    winner = "claim_a"
    source_a = str(conflict.get("source_a", ""))
    source_b = str(conflict.get("source_b", ""))
    if source_b and not source_a:
        winner = "claim_b"

    return {
        "entity": conflict.get("entity", "unknown"),
        "winning_claim": conflict.get(winner, ""),
        "winning_source": conflict.get("source_a" if winner == "claim_a" else "source_b", ""),
        "reasoning": "Selected the claim with available source support; full LLM arbitration is reserved for live mode.",
    }


def route_after_arbitrator(state: QueryMindState) -> str:
    retry_count = state.get("retry_count", 0)
    if state.get("conflicts_detected") and retry_count < 1:
        return "synthesize"
    return "synthesize"


def _first_source(result: dict[str, Any]) -> str:
    citations = result.get("citations") or []
    if not citations:
        return ""
    return str(citations[0].get("url") or citations[0].get("source") or "")


def _boolean_conflict(left: str, right: str) -> bool:
    left_lower = left.lower()
    right_lower = right.lower()
    positive = {" is ", " are ", " can ", " available ", " supported "}
    negative = {" is not ", " are not ", " cannot ", " unavailable ", " unsupported ", " discontinued "}
    return any(term in left_lower for term in positive) and any(term in right_lower for term in negative)


def _shared_entity(left: str, right: str) -> str:
    left_entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", left))
    right_entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", right))
    shared = sorted(left_entities & right_entities)
    return shared[0] if shared else ""


def _now_ms() -> int:
    return int(time.time() * 1000)
