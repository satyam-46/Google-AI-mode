"""FastAPI server exposing the Phase 2 LangGraph QueryMind runtime."""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import json
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core.memory.session_store import SessionStore
from graph.query_mind_graph import (
    get_query_graph_state,
    graph_config,
    resume_query_graph,
    run_query_graph,
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


class ResumeRequest(BaseModel):
    feedback: dict[str, Any] = Field(default_factory=dict)


async def _run_graph_and_stream(query: str, session_id: str, token_queue: asyncio.Queue):
    """Run the LangGraph pipeline and push token strings to `token_queue`."""
    try:
        state = await run_query_graph(query=query, session_id=session_id)
        await _session_store.save(session_id, state)
        answer_text = state.get("streaming_answer") or state.get("final_answer", {}).get("answer_text", "")
        if not answer_text:
            await token_queue.put(None)
            return

        tokens = answer_text.split()
        for tok in tokens:
            await token_queue.put(tok)
            await asyncio.sleep(0.02)
    except Exception as exc:
        state = {"original_query": query, "session_id": session_id, "error": str(exc)}
        await _session_store.save(session_id, state)
        await token_queue.put(json.dumps({"error": str(exc)}))
    finally:
        await token_queue.put(None)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    session_id = payload.session_id or str(uuid.uuid4())
    state = await run_query_graph(query=payload.query, session_id=session_id)
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


@app.post("/query/stream")
async def stream_query(payload: QueryRequest):
    session_id = payload.session_id or str(uuid.uuid4())
    token_queue: asyncio.Queue = asyncio.Queue()

    asyncio.create_task(_run_graph_and_stream(payload.query, session_id, token_queue))

    async def event_generator():
        while True:
            token = await token_queue.get()
            if token is None:
                yield "data: [DONE]\n\n"
                break
            # SSE data frame
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/query/{session_id}/state")
async def get_query_state(session_id: str):
    checkpoint = get_query_graph_state(session_id)
    state = checkpoint.values if checkpoint else await _session_store.load(session_id)
    if not state:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return {"config": graph_config(session_id), "state": state, "next": getattr(checkpoint, "next", ())}


@app.post("/query/{session_id}/resume")
async def resume_with_human_feedback(session_id: str, payload: ResumeRequest):
    state = await resume_query_graph(session_id=session_id, feedback=payload.feedback or {"approved": True})
    await _session_store.save(session_id, state)
    return state
