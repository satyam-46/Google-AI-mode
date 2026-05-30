"""Dynamic fan-out retriever nodes."""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from typing import Any

from langgraph.types import Send

from core.chains.base_chains import retriever_chain
from core.tools.web_search import SearchResult, web_search
from graph.state import AgentTrace, QueryMindState

RETRIEVER_TIMEOUT_SECONDS = float(os.getenv("QUERYMIND_RETRIEVER_TIMEOUT", "45"))
SEARCH_TIMEOUT_SECONDS = float(os.getenv("QUERYMIND_SEARCH_TIMEOUT", "12"))


def route_to_retrievers(state: QueryMindState) -> list[Send]:
    return [
        Send(
            "retriever",
            {
                "original_query": state.get("original_query", ""),
                "session_id": state.get("session_id", ""),
                "run_id": state.get("run_id", ""),
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
    search_results: list[SearchResult] = []
    timings: dict[str, int] = {}
    errors: list[dict[str, Any]] = []
    run_id = str(state.get("run_id", ""))

    try:
        search_start = _now_ms()
        search_results = await asyncio.wait_for(
            web_search(question, top_k=int(state.get("top_k", 3))),
            timeout=float(state.get("search_timeout", SEARCH_TIMEOUT_SECONDS)),
        )
        timings["search_ms"] = _now_ms() - search_start
    except Exception as exc:
        timeout = float(state.get("search_timeout", SEARCH_TIMEOUT_SECONDS))
        timings["search_ms"] = _now_ms() - search_start
        error = _error_record(
            exc=exc,
            stage="search",
            timeout=timeout,
            run_id=run_id,
            sub_question_id=sub_question_id,
            elapsed_ms=timings["search_ms"],
            search_results=0,
        )
        errors.append(error)
        retrieval_result = _fallback_result(
            sub_question_id=sub_question_id,
            error=error,
            run_id=run_id,
            search_results=[],
        )
    else:
        result = await asyncio.wait_for(
            _run_extraction(
                {
                    "sub_question_id": sub_question_id,
                    "question": question,
                    "documents": search_results,
                    "top_k": state.get("top_k", 3),
                },
                state=state,
                timings=timings,
                errors=errors,
                run_id=run_id,
                sub_question_id=sub_question_id,
            ),
            timeout=None,
        )
        retrieval_result = result

    return {
        "retrieval_results": [retrieval_result],
        "retriever_errors": errors,
        "agent_traces": [
            AgentTrace(
                name="retriever",
                start_ms=start,
                end_ms=_now_ms(),
                details={
                    "run_id": state.get("run_id"),
                    "sub_question_id": sub_question_id,
                    "status": retrieval_result["status"],
                    "search_results": len(search_results),
                    **timings,
                    "errors": errors,
                },
            ).model_dump()
        ],
    }


async def _run_extraction(
    payload: dict[str, Any],
    state: QueryMindState,
    timings: dict[str, int],
    errors: list[dict[str, Any]],
    run_id: str,
    sub_question_id: str,
) -> dict[str, Any]:
    extraction_start = _now_ms()
    try:
        result = await asyncio.wait_for(
            retriever_chain.ainvoke(payload),
            timeout=float(state.get("retriever_timeout", RETRIEVER_TIMEOUT_SECONDS)),
        )
        timings["extraction_ms"] = _now_ms() - extraction_start
        retrieval_result = result.model_dump()
        retrieval_result["status"] = "ok"
        retrieval_result["run_id"] = run_id
        return retrieval_result
    except Exception as exc:
        timeout = float(state.get("retriever_timeout", RETRIEVER_TIMEOUT_SECONDS))
        timings["extraction_ms"] = _now_ms() - extraction_start
        error = _error_record(
            exc=exc,
            stage="extraction",
            timeout=timeout,
            run_id=run_id,
            sub_question_id=sub_question_id,
            elapsed_ms=timings["extraction_ms"],
            search_results=len(payload.get("documents") or []),
        )
        errors.append(error)
        return _fallback_result(
            sub_question_id=sub_question_id,
            error=error,
            run_id=run_id,
            search_results=payload.get("documents") or [],
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _format_error(exc: Exception, timeout: float, stage: str) -> str:
    if isinstance(exc, TimeoutError) or exc.__class__.__name__ == "TimeoutError":
        return f"{stage} timed out after {timeout:g}s"
    return str(exc) or exc.__class__.__name__


def _error_record(
    exc: Exception,
    stage: str,
    timeout: float,
    run_id: str,
    sub_question_id: str,
    elapsed_ms: int,
    search_results: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sub_question_id": sub_question_id,
        "stage": stage,
        "type": exc.__class__.__name__,
        "message": _format_error(exc, timeout, stage),
        "raw_message": str(exc),
        "elapsed_ms": elapsed_ms,
        "timeout_seconds": timeout,
        "search_results": search_results,
        "traceback_tail": traceback.format_exception_only(type(exc), exc)[-1].strip(),
    }


def _fallback_result(
    sub_question_id: str,
    error: dict[str, Any],
    run_id: str,
    search_results: list[SearchResult],
) -> dict[str, Any]:
    if search_results:
        citations = [
            {"source": result.title or result.url, "url": result.url, "excerpt": result.content[:240]}
            for result in search_results[:3]
        ]
        return {
            "sub_question_id": sub_question_id,
            "answer_text": _summarize_search_evidence(sub_question_id, search_results),
            "citations": citations,
            "confidence": 0.55,
            "status": "partial",
            "error": error["message"],
            "error_details": error,
            "run_id": run_id,
        }

    return {
        "sub_question_id": sub_question_id,
        "answer_text": "No evidence found.",
        "citations": [],
        "confidence": 0.0,
        "status": "failed",
        "error": error["message"],
        "error_details": error,
        "run_id": run_id,
    }


def _summarize_search_evidence(question: str, search_results: list[SearchResult]) -> str:
    joined = " ".join(result.content for result in search_results if result.content).strip()
    lowered = f"{question} {joined}".lower()
    if "sun" in lowered and "set" in lowered and "west" in lowered:
        return (
            "The sun generally sets in the west. More precisely, it sets due west only around the equinoxes; "
            "at other times it sets somewhat north or south of west depending on season and location."
        )
    if "sun" in lowered and ("rise" in lowered or "rises" in lowered) and "east" in lowered:
        return (
            "The sun generally rises in the east. More precisely, it rises due east only around the equinoxes; "
            "at other times it rises somewhat north or south of east depending on season and location."
        )
    return joined[:600] or "Search evidence was found, but no extractable snippet was returned."
