"""Retriever agent node."""

from __future__ import annotations

import asyncio
from typing import Any

from core.chains.base_chains import retriever_chain


async def retriever_node(state: dict[str, Any]) -> dict[str, Any]:
    """Retrieve evidence for each planner sub-question in parallel."""
    sub_questions = state.get("sub_questions") or []
    tasks = [
        retriever_chain.ainvoke(
            {
                "sub_question_id": sub_question.get("id", ""),
                "question": sub_question.get("question", ""),
                "top_k": state.get("top_k", 3),
            }
        )
        for sub_question in sub_questions
    ]

    if not tasks:
        return {"retrieval_results": []}

    results = await asyncio.gather(*tasks, return_exceptions=True)
    retrieval_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
            continue
        retrieval_results.append(result.model_dump())

    output: dict[str, Any] = {"retrieval_results": retrieval_results}
    if errors:
        output["retriever_errors"] = errors
    return output
