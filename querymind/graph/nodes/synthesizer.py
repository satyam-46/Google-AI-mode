"""Streaming synthesizer node."""

from __future__ import annotations

import time
from typing import Any

from core.chains.base_chains import synthesizer_chain
from graph.state import AgentTrace, QueryMindState


async def synthesizer_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    retrieval_results = state.get("retrieval_results", [])
    usable_results = [
        item
        for item in retrieval_results
        if item.get("status") != "failed" and float(item.get("confidence", 0.0)) >= 0.5
    ]
    confidence_marker = "[HIGH CONFIDENCE]" if usable_results else "[LOW CONFIDENCE]"
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
                details={"evidence_items": len(usable_results)},
            ).model_dump()
        ],
    }


def _now_ms() -> int:
    return int(time.time() * 1000)
