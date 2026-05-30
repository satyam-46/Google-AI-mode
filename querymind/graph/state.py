"""Shared state schema for the Phase 2 LangGraph orchestration layer."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


class QueryMindState(TypedDict, total=False):
    original_query: str
    session_id: str
    started_at_ms: int

    sub_question: dict[str, Any]
    sub_questions: list[dict[str, Any]]
    complexity_score: float

    retrieval_results: Annotated[list[dict[str, Any]], operator.add]
    retriever_errors: Annotated[list[str], operator.add]

    conflicts_detected: list[dict[str, Any]]
    arbitration_results: list[dict[str, Any]]
    retry_count: int

    streaming_answer: str
    final_answer: dict[str, Any]

    confidence_score: dict[str, Any]
    requires_human_review: bool
    human_feedback: dict[str, Any]

    cache_hits: list[str]
    session_history: list[dict[str, Any]]

    agent_traces: Annotated[list[dict[str, Any]], operator.add]
    total_tokens_used: int
    total_latency_ms: int


class AgentTrace(BaseModel):
    name: str
    start_ms: int
    end_ms: int
    tokens_used: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


def example_state() -> QueryMindState:
    return {
        "original_query": "",
        "session_id": "",
        "retrieval_results": [],
        "retriever_errors": [],
        "agent_traces": [],
        "cache_hits": [],
        "total_tokens_used": 0,
        "total_latency_ms": 0,
    }
