"""Dynamic fan-out retriever nodes."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langgraph.types import Send

from core.chains.base_chains import retriever_chain
from graph.state import AgentTrace, QueryMindState

RETRIEVER_TIMEOUT_SECONDS = 15


def route_to_retrievers(state: QueryMindState) -> list[Send]:
    return [
        Send(
            "retriever",
            {
                "original_query": state.get("original_query", ""),
                "session_id": state.get("session_id", ""),
                "sub_question": sub_question,
                "top_k": state.get("top_k", 3),
            },
        )
        for sub_question in state.get("sub_questions", [])
    ]


async def retriever_node(state: QueryMindState) -> dict[str, Any]:
    start = _now_ms()
    sub_question = state.get("sub_question", {})
    sub_question_id = str(sub_question.get("id", ""))
    question = str(sub_question.get("question") or sub_question.get("search_query") or "")

    try:
        result = await asyncio.wait_for(
            retriever_chain.ainvoke(
                {
                    "sub_question_id": sub_question_id,
                    "question": question,
                    "top_k": state.get("top_k", 3),
                }
            ),
            timeout=float(state.get("retriever_timeout", RETRIEVER_TIMEOUT_SECONDS)),
        )
        retrieval_result = result.model_dump()
        retrieval_result["status"] = "ok"
        errors: list[str] = []
    except Exception as exc:
        retrieval_result = {
            "sub_question_id": sub_question_id,
            "answer_text": "No evidence found.",
            "citations": [],
            "confidence": 0.0,
            "status": "failed",
            "error": str(exc),
        }
        errors = [f"{sub_question_id}: {exc}"]

    return {
        "retrieval_results": [retrieval_result],
        "retriever_errors": errors,
        "agent_traces": [
            AgentTrace(
                name="retriever",
                start_ms=start,
                end_ms=_now_ms(),
                details={"sub_question_id": sub_question_id, "status": retrieval_result["status"]},
            ).model_dump()
        ],
    }


def _now_ms() -> int:
    return int(time.time() * 1000)
