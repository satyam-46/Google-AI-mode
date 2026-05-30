"""Streaming synthesizer node."""

from __future__ import annotations

import time
from typing import Any

from core.chains.base_chains import synthesizer_chain
from graph.state import AgentTrace, QueryMindState


async def synthesizer_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    retrieval_results = _current_run_items(state.get("retrieval_results", []), state.get("run_id", ""))
    usable_results = [
        item
        for item in retrieval_results
        if item.get("status") != "failed" and float(item.get("confidence", 0.0)) >= 0.5
    ]
    confidence_marker = _confidence_marker(usable_results)
    final_answer = await synthesizer_chain.ainvoke(
        {
            "query": state.get("original_query", ""),
            "evidence": usable_results or retrieval_results,
        }
    )
    answer = final_answer.model_dump()
    answer["answer_text"] = f"{confidence_marker} {answer['answer_text']}"

    return {
        "streaming_answer": answer["answer_text"],
        "final_answer": answer,
        "agent_traces": [
            AgentTrace(
                name="synthesizer",
                start_ms=start,
                end_ms=_now_ms(),
                details={"run_id": state.get("run_id"), "evidence_items": len(usable_results)},
            ).model_dump()
        ],
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _current_run_items(items: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    if not run_id:
        return items
    current = [item for item in items if item.get("run_id") == run_id]
    return current or [item for item in items if not item.get("run_id")]


def _confidence_marker(usable_results: list[dict[str, Any]]) -> str:
    if not usable_results:
        return "[LOW CONFIDENCE]"
    average_confidence = sum(float(item.get("confidence", 0.0)) for item in usable_results) / len(usable_results)
    has_partial = any(item.get("status") == "partial" for item in usable_results)
    return "[HIGH CONFIDENCE]" if average_confidence >= 0.75 and not has_partial else "[LOW CONFIDENCE]"
