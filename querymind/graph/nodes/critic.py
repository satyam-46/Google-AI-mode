"""Critic node with human-in-the-loop interrupt support."""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from core.chains.base_chains import critic_chain
from graph.state import AgentTrace, QueryMindState

SENSITIVE_TOPICS = {"medical", "legal", "financial"}


async def critic_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    confidence = await critic_chain.ainvoke({"answer": state.get("final_answer", {})})
    needs_review = confidence.score < 0.6 or bool(SENSITIVE_TOPICS & set(confidence.flags))

    if needs_review:
        feedback = interrupt(
            {
                "reason": "low_confidence_or_sensitive_topic",
                "answer": state.get("final_answer", {}),
                "score": confidence.model_dump(),
            }
        )
        if isinstance(feedback, dict) and feedback.get("approved") is False:
            needs_review = True
        else:
            needs_review = False

    return {
        "confidence_score": confidence.model_dump(),
        "requires_human_review": needs_review,
        "human_feedback": feedback if "feedback" in locals() else state.get("human_feedback", {}),
        "agent_traces": [
            AgentTrace(
                name="critic",
                start_ms=start,
                end_ms=_now_ms(),
                details={"requires_human_review": needs_review, "flags": confidence.flags},
            ).model_dump()
        ],
    }


def route_after_critic(state: QueryMindState) -> str:
    return "approved"


def _now_ms() -> int:
    return int(time.time() * 1000)
