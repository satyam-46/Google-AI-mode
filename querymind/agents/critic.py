"""Critic agent node."""

from __future__ import annotations

from typing import Any

from core.chains.base_chains import critic_chain


async def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    """Score the final answer for support and review risk."""
    confidence = await critic_chain.ainvoke({"answer": state.get("final_answer", {})})
    return {
        "confidence_score": confidence.model_dump(),
        "requires_human_review": confidence.score < 0.6 or bool(confidence.flags),
    }
