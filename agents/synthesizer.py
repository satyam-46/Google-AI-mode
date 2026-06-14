"""Synthesizer agent node."""

from __future__ import annotations

from typing import Any

from core.chains.base_chains import synthesizer_chain


async def synthesizer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Synthesize retrieved evidence into a final cited answer."""
    final_answer = await synthesizer_chain.ainvoke(
        {
            "query": state.get("original_query", ""),
            "evidence": state.get("retrieval_results", []),
        }
    )
    return {"final_answer": final_answer.model_dump()}
