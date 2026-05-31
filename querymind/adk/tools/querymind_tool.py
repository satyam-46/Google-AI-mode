"""ADK FunctionTool wrapper around the LangGraph QueryMind runtime."""

from __future__ import annotations

from typing import Any

from graph.query_mind_graph import run_query_graph


async def run_querymind(query: str, session_id: str) -> dict[str, Any]:
    """Run QueryMind for deep, multi-source research questions requiring synthesis."""
    state = await run_query_graph(query=query, session_id=session_id)
    final_answer = state.get("final_answer", {})
    return {
        "answer": final_answer.get("answer_text", ""),
        "citations": final_answer.get("citations", []),
        "confidence": final_answer.get("confidence", 0.0),
        "requires_human_review": bool(state.get("requires_human_review", False)),
        "state": state,
    }


try:
    from google.adk.tools import FunctionTool

    querymind_tool = FunctionTool(run_querymind)
except Exception:  # pragma: no cover - keeps imports usable if ADK changes.
    querymind_tool = None


__all__ = ["querymind_tool", "run_querymind"]

