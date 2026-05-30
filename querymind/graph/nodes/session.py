"""Run initialization node."""

from __future__ import annotations

import time
import uuid
from typing import Any

from graph.state import QueryMindState


async def initialize_node(state: QueryMindState) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    return {
        "run_id": run_id,
        "started_at_ms": int(time.time() * 1000),
        "cache_hits": [],
        "retry_count": 0,
        "session_history": _append_history(state),
    }


def _append_history(state: QueryMindState) -> list[dict[str, Any]]:
    history = list(state.get("session_history", []))
    final_answer = state.get("final_answer")
    previous_query = state.get("original_query")
    previous_run_id = state.get("run_id")
    if not final_answer or not previous_query:
        return history
    if history and history[-1].get("run_id") == previous_run_id:
        return history

    history.append(
        {
            "run_id": previous_run_id,
            "query": previous_query,
            "answer_summary": str(final_answer.get("answer_text", ""))[:300],
            "final_answer": final_answer,
            "completed_at_ms": int(time.time() * 1000),
        }
    )
    return history[-20:]
