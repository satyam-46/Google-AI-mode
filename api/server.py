"""FastAPI server exposing the QueryMind runtime."""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import json
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from adk.orchestrator import route_query
from core.memory.session_store import SessionStore
from core.hardening import (
    BudgetExceededError,
    CircuitOpenError,
    RateLimitExceededError,
    get_circuit_breaker,
    get_cost_controller,
    get_dead_letter_store,
    get_rate_limiter,
)
from graph.query_mind_graph import (
    get_projected_query_graph_state,
    get_query_graph_state,
    project_current_run_state,
    graph_config,
    resume_query_graph,
    run_query_graph,
    stream_query_graph_events,
)

load_dotenv()

app = FastAPI(title="QueryMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_store = SessionStore()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question to answer.")
    session_id: str | None = Field(None, description="Optional session id for state lookup.")


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    requires_human_review: bool = False
    state: dict[str, Any]


class OrchestratedQueryResponse(BaseModel):
    session_id: str
    route: str
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    requires_human_review: bool = False
    route_decision: dict[str, Any]
    state: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    feedback: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    stage: str
    type: str
    message: str


class StreamEvent(BaseModel):
    event: Literal["node_start", "node_end", "token", "error", "graph_event"]
    session_id: str
    sequence: int
    node: str | None = None
    token: str | None = None
    error: ErrorPayload | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class QueryStateResponse(BaseModel):
    session_id: str
    config: dict[str, Any]
    state: dict[str, Any]
    next: list[str] = Field(default_factory=list)


class ResumeResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    requires_human_review: bool = False
    state: dict[str, Any]


async def _run_graph_and_stream(query: str, session_id: str, token_queue: asyncio.Queue):
    """Run the LangGraph pipeline and push token strings to `token_queue`."""
    try:
        async for event in stream_query_graph_events(query=query, session_id=session_id):
            await token_queue.put(event)
        state = get_projected_query_graph_state(session_id)
        if state:
            await _session_store.save(session_id, state)
        get_circuit_breaker().record_success()
    except Exception as exc:
        get_circuit_breaker().record_failure()
        state = {"original_query": query, "session_id": session_id, "error": str(exc)}
        await _session_store.save(session_id, state)
        await token_queue.put(
            {
                "event": "error",
                "error": {
                    "stage": "stream",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "session_id": session_id,
            }
        )
    finally:
        await token_queue.put(None)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request):
    session_id = payload.session_id or str(uuid.uuid4())
    guard = enforce_request_controls(payload.query, session_id, request)
    if guard is not None:
        return guard
    try:
        state = await run_query_graph(query=payload.query, session_id=session_id)
        get_circuit_breaker().record_success()
    except Exception as exc:
        get_circuit_breaker().record_failure()
        await get_dead_letter_store().record_failure(
            session_id=session_id,
            query=payload.query,
            stage="api_query",
            error=exc,
            state={"session_id": session_id},
        )
        return hardening_error_response("api_query", exc, status_code=500)
    await _session_store.save(session_id, state)
    final_answer = state.get("final_answer", {})
    return QueryResponse(
        session_id=session_id,
        answer=final_answer.get("answer_text", ""),
        citations=final_answer.get("citations", []),
        confidence=final_answer.get("confidence", 0.0),
        requires_human_review=bool(state.get("requires_human_review", False)),
        state=state,
    )


@app.post("/query/orchestrated", response_model=OrchestratedQueryResponse)
async def orchestrated_query(payload: QueryRequest, request: Request):
    session_id = payload.session_id or str(uuid.uuid4())
    guard = enforce_request_controls(payload.query, session_id, request)
    if guard is not None:
        return guard
    prior_state = await _session_store.load(session_id)
    try:
        result = await route_query(
            query=payload.query,
            session_id=session_id,
            has_session_context=bool(prior_state),
            session_context=prior_state,
        )
        if result.get("error"):
            get_circuit_breaker().record_failure()
        else:
            get_circuit_breaker().record_success()
    except Exception as exc:
        get_circuit_breaker().record_failure()
        await get_dead_letter_store().record_failure(
            session_id=session_id,
            query=payload.query,
            stage="api_orchestrated",
            error=exc,
            state={"session_id": session_id},
        )
        return hardening_error_response("api_orchestrated", exc, status_code=500)
    if result.get("state"):
        await _session_store.save(session_id, result["state"])
    return OrchestratedQueryResponse(
        session_id=session_id,
        route=result.get("route", ""),
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        confidence=result.get("confidence", 0.0),
        requires_human_review=bool(result.get("requires_human_review", False)),
        route_decision=result.get("route_decision", {}),
        state=result.get("state", {}),
        error=result.get("error"),
    )


@app.post("/query/stream")
async def stream_query(payload: QueryRequest, request: Request):
    session_id = payload.session_id or str(uuid.uuid4())
    guard = enforce_request_controls(payload.query, session_id, request)
    if guard is not None:
        return guard
    token_queue: asyncio.Queue = asyncio.Queue()

    asyncio.create_task(_run_graph_and_stream(payload.query, session_id, token_queue))

    async def event_generator():
        sequence = 0
        while True:
            event = await token_queue.get()
            if event is None:
                yield "event: done\ndata: [DONE]\n\n"
                break
            sequence += 1
            normalized = normalize_stream_event(event, session_id=session_id, sequence=sequence)
            yield format_sse_event(normalized)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/query/{session_id}/state", response_model=QueryStateResponse)
async def get_query_state(session_id: str):
    checkpoint = get_query_graph_state(session_id)
    checkpoint_values = dict(getattr(checkpoint, "values", {}) or {}) if checkpoint else {}
    state = project_current_run_state(checkpoint_values) if checkpoint_values else await _session_store.load(session_id)
    if not state:
        return JSONResponse(
            {
                "error": {
                    "stage": "state_lookup",
                    "type": "NotFound",
                    "message": f"No state found for session_id={session_id}",
                }
            },
            status_code=404,
        )
    return QueryStateResponse(
        session_id=session_id,
        config=graph_config(session_id),
        state=state,
        next=list(getattr(checkpoint, "next", ()) or ()),
    )


@app.post("/query/{session_id}/resume", response_model=ResumeResponse)
async def resume_with_human_feedback(session_id: str, payload: ResumeRequest):
    try:
        state = await resume_query_graph(session_id=session_id, feedback=payload.feedback or {"approved": True})
    except Exception as exc:
        return JSONResponse(
            {
                "error": {
                    "stage": "resume",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            },
            status_code=409,
        )
    await _session_store.save(session_id, state)
    final_answer = state.get("final_answer", {})
    return ResumeResponse(
        session_id=session_id,
        answer=final_answer.get("answer_text", ""),
        citations=final_answer.get("citations", []),
        confidence=final_answer.get("confidence", 0.0),
        requires_human_review=bool(state.get("requires_human_review", False)),
        state=state,
    )


def normalize_stream_event(event: dict[str, Any], session_id: str, sequence: int) -> dict[str, Any]:
    """Convert graph stream events into the public SSE schema."""
    raw_event = str(event.get("event") or "graph_event")
    if raw_event == "on_chain_start":
        return StreamEvent(
            event="node_start",
            session_id=session_id,
            sequence=sequence,
            node=str(event.get("node") or ""),
            data={"raw_event": raw_event},
        ).model_dump(exclude_none=True)
    if raw_event == "on_chain_end":
        return StreamEvent(
            event="node_end",
            session_id=session_id,
            sequence=sequence,
            node=str(event.get("node") or ""),
            data={"raw_event": raw_event},
        ).model_dump(exclude_none=True)
    if raw_event == "token":
        return StreamEvent(
            event="token",
            session_id=session_id,
            sequence=sequence,
            token=str(event.get("token") or ""),
            data={"raw_event": raw_event},
        ).model_dump(exclude_none=True)
    if raw_event == "error":
        error = event.get("error")
        if not isinstance(error, dict):
            error = {"stage": "stream", "type": "RuntimeError", "message": str(error)}
        return StreamEvent(
            event="error",
            session_id=session_id,
            sequence=sequence,
            error=ErrorPayload(
                stage=str(error.get("stage") or "stream"),
                type=str(error.get("type") or "RuntimeError"),
                message=str(error.get("message") or ""),
            ),
            data={"raw_event": raw_event},
        ).model_dump(exclude_none=True)
    return StreamEvent(
        event="graph_event",
        session_id=session_id,
        sequence=sequence,
        node=str(event.get("node") or "") or None,
        data={**event, "raw_event": raw_event},
    ).model_dump(exclude_none=True)


def format_sse_event(event: dict[str, Any]) -> str:
    """Format one public stream event as a Server-Sent Event frame."""
    return f"event: {event['event']}\ndata: {json.dumps(event, ensure_ascii=True)}\n\n"


def enforce_request_controls(query_text: str, session_id: str, request: Request) -> JSONResponse | None:
    """Apply rate, circuit, and spend controls before expensive work starts."""
    user_id = request.headers.get("x-user-id") or session_id
    try:
        if not get_rate_limiter().allow(user_id):
            raise RateLimitExceededError(f"Rate limit exceeded for user {user_id}")
        if not get_circuit_breaker().allow_request():
            raise CircuitOpenError("Global upstream circuit breaker is open")
        preflight = get_cost_controller().preflight(query_text)
        get_cost_controller().record_spend(float(preflight["estimated_cost_usd"]))
    except RateLimitExceededError as exc:
        return hardening_error_response("rate_limit", exc, status_code=429)
    except CircuitOpenError as exc:
        return hardening_error_response("circuit_breaker", exc, status_code=503)
    except BudgetExceededError as exc:
        return hardening_error_response("cost_control", exc, status_code=402)
    return None


def hardening_error_response(stage: str, error: Exception, status_code: int) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            }
        },
        status_code=status_code,
    )
