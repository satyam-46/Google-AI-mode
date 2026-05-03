"""FastAPI server exposing a Server-Sent Events streaming endpoint for QueryMind.

This is a lightweight wiring that runs the local chain stubs in background
and streams synthesized tokens to the client via SSE. It also saves simple
session state in `core.memory.session_store.SessionStore` for inspection
and resume hooks.
"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import json
import time
from typing import Any

from core.chains.base_chains import planner_chain, retriever_chain, synthesizer_chain
from core.memory.session_store import SessionStore

app = FastAPI(title="QueryMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_store = SessionStore()


async def _run_graph_and_stream(query: str, session_id: str, token_queue: asyncio.Queue):
    """Run the simple chain pipeline and push token strings to `token_queue`.

    This function is intentionally simple: it calls planner_chain -> parallel
    retrievers -> synthesizer_chain then streams the final answer token-by-token.
    """
    start = time.time()
    state: dict[str, Any] = {"original_query": query, "session_id": session_id, "agent_traces": []}
    try:

        # Planner
        p_start = time.time()
        planner_out = await planner_chain.ainvoke({"query": query})
        p_end = time.time()
        state["sub_questions"] = [item.model_dump() for item in planner_out.sub_questions]
        state["agent_traces"].append({"name": "planner", "start_ms": int(p_start * 1000), "end_ms": int(p_end * 1000)})

        # Fan-out retrievers (simple parallel gather)
        sub_qs = state.get("sub_questions", [])
        retriever_tasks = []
        for sq in sub_qs:
            retriever_tasks.append(
                asyncio.create_task(
                    retriever_chain.ainvoke(
                        {
                            "sub_question_id": sq.get("id", ""),
                            "question": sq.get("question", ""),
                            "top_k": 3,
                        }
                    )
                )
            )

        retriever_results = []
        if retriever_tasks:
            done, pending = await asyncio.wait(retriever_tasks, timeout=15)
            for t in done:
                try:
                    retriever_results.append(t.result())
                except Exception:
                    retriever_results.append(None)
            for t in pending:
                t.cancel()

        retrieval_results = [result for result in retriever_results if result is not None]
        state["retrieval_results"] = [result.model_dump() for result in retrieval_results]

        # Synthesize
        synth_start = time.time()
        synth_out = await synthesizer_chain.ainvoke({"query": query, "evidence": retrieval_results})
        synth_end = time.time()
        state["final_answer"] = synth_out.model_dump()
        state["agent_traces"].append({"name": "synthesizer", "start_ms": int(synth_start * 1000), "end_ms": int(synth_end * 1000)})

        # Save state
        state["total_latency_ms"] = int((time.time() - start) * 1000)
        await _session_store.save(session_id, state)

        # Stream the answer token-by-token
        answer_text = synth_out.answer_text
        if not answer_text:
            await token_queue.put(None)
            return

        # naive token streaming: split on whitespace and yield small chunks
        tokens = answer_text.split()
        for tok in tokens:
            await token_queue.put(tok)
            await asyncio.sleep(0.02)
    except Exception as exc:
        state["error"] = str(exc)
        await _session_store.save(session_id, state)
        await token_queue.put(json.dumps({"error": str(exc)}))
    finally:
        await token_queue.put(None)


@app.post("/query/stream")
async def stream_query(request: Request):
    payload = await request.json()
    query = payload.get("query") or payload.get("q")
    if not query:
        return JSONResponse({"error": "missing query"}, status_code=400)

    session_id = payload.get("session_id") or str(uuid.uuid4())
    token_queue: asyncio.Queue = asyncio.Queue()

    asyncio.create_task(_run_graph_and_stream(query, session_id, token_queue))

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
    state = await _session_store.load(session_id)
    if not state:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return state


@app.post("/query/{session_id}/resume")
async def resume_with_human_feedback(session_id: str, request: Request):
    payload = await request.json()
    # For now, simply attach feedback to session state and return it
    state = await _session_store.load(session_id)
    if not state:
        return JSONResponse({"error": "not_found"}, status_code=404)
    state["human_feedback"] = payload
    await _session_store.save(session_id, state)
    return state
