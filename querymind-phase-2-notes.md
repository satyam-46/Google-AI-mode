# QueryMind Phase 2 Notes

## Current Project Stage

QueryMind is now in the **Phase 2 LangGraph orchestration** stage.

Phase 1 built independently testable LangChain tools and chains. Phase 2 wraps those pieces in a real `StateGraph` so the project behaves like a multi-agent search system:

- Session initialization
- Cache lookup before planning
- Adaptive query decomposition
- Dynamic parallel retriever fan-out
- Fan-in through reducer-backed shared state
- Conflict detection and arbitration
- Grounded answer synthesis
- Critic scoring
- Human-in-the-loop interrupt and resume
- Cache persistence after a completed run
- FastAPI endpoints backed by the graph runtime

The graph is still intentionally runnable without live API keys. In local test mode, deterministic fake LLM fallbacks and empty Tavily results let the full orchestration path run without network calls.

## Current Flow

```text
User query
   |
   v
initialize
   |
   v
cache_lookup
   |
   |-- cache hit --> synthesizer
   |
   v
planner
   |
   v
route_to_retrievers with Send(...)
   |
   v
parallel retriever branches
   |
   v
arbitrator
   |
   v
synthesizer
   |
   v
critic
   |
   |-- low confidence / sensitive topic --> interrupt for human review
   |
   v
cache_store
   |
   v
END
```

## Why LangGraph Now

LangGraph is used in Phase 2 because the system has moved beyond simple chain composition.

LangGraph gives QueryMind:

- A typed global state object
- Reducers for merging parallel branch outputs
- Dynamic routing with `Send`
- Checkpointed runs using `thread_id`
- Runtime state inspection with `graph.get_state(config)`
- Human review pauses with `interrupt()`
- Resume support with `Command(resume=...)`

Interview defense:

> I kept LangChain responsible for tools, prompts, parsers, and model calls, then introduced LangGraph only when orchestration became stateful. LangGraph owns fan-out, fan-in, retries, checkpointing, and human-in-the-loop control flow.

## Important Files

### `graph/state.py`

This defines `QueryMindState`, the shared state passed between graph nodes.

Important Phase 2 change:

```python
retrieval_results: Annotated[list[dict[str, Any]], operator.add]
agent_traces: Annotated[list[dict[str, Any]], operator.add]
```

Why this matters:

> Parallel retriever branches return separate partial updates. Reducers let LangGraph merge those lists instead of letting branches overwrite each other.

### `graph/query_mind_graph.py`

This assembles the full `StateGraph`.

It wires:

- `initialize`
- `cache_lookup`
- `planner`
- `retriever`
- `arbitrator`
- `synthesizer`
- `critic`
- `cache_store`

It also exposes helper functions:

- `build_graph()`
- `run_query_graph()`
- `resume_query_graph()`
- `get_query_graph_state()`
- `project_current_run_state()`
- `stream_query_graph_events()`
- `graph_config(session_id)`

Why this matters:

> The API no longer owns orchestration logic. It now calls a compiled graph, which is inspectable, resumable, and checkpointed.

Important debugging lesson:

> LangGraph checkpoint state is session-scoped by `thread_id`. Reducer-backed fields such as `retrieval_results` and `agent_traces` are append-oriented, so old branches can remain in the checkpoint across multiple requests for the same session. The API therefore returns a projected current-run view filtered by `run_id`, while keeping previous completed turns in `session_history`.

### `graph/nodes/planner.py`

This calls the Phase 1 `planner_chain` and computes a simple complexity score.

The score considers:

- Number of sub-questions
- Capitalized entities
- Comparison language
- Temporal language
- Causal language
- Query length

The complexity score caps fan-out:

- Low complexity: up to 2 sub-questions
- Medium complexity: up to 5
- High complexity: up to 8

### `graph/nodes/retriever.py`

This implements the core LangGraph fan-out pattern.

```python
Send("retriever", {"sub_question": sub_question, ...})
```

Each retriever branch:

- Runs independently
- Has stage-specific timeouts
- Runs web search before LLM extraction
- Preserves search snippets as partial evidence if extraction times out
- Returns structured error records instead of raising
- Appends its own trace entry

Why this matters:

> One failing retriever should degrade the answer, not crash or hang the graph.

Timeout configuration:

```bash
QUERYMIND_SEARCH_TIMEOUT=12
QUERYMIND_RETRIEVER_TIMEOUT=45
```

The search timeout bounds Tavily/web retrieval. The retriever timeout bounds the LLM extraction step after search results are available.

Debugging lesson:

> A timeout caused by `asyncio.wait_for(...)` raises Python's own `TimeoutError`. If the Gemini call is cancelled before it returns, there may be no provider-level Gemini exception to capture. The truthful error is that the extraction stage exceeded its timeout. To make this diagnosable, retriever errors include stage, exception type, raw message, elapsed time, timeout threshold, search-result count, and traceback tail.

Example retriever error:

```json
{
  "run_id": "...",
  "sub_question_id": "q1",
  "stage": "extraction",
  "type": "TimeoutError",
  "message": "extraction timed out after 45s",
  "raw_message": "",
  "elapsed_ms": 45003,
  "timeout_seconds": 45,
  "search_results": 3,
  "traceback_tail": "TimeoutError"
}
```

Partial-evidence fallback:

> If search succeeds but extraction times out, the result becomes `status="partial"` instead of `failed`. This keeps useful citations and snippets alive while honestly lowering confidence.

For common sun rise/set questions, the fallback condenses noisy snippets into a concise statement such as:

```text
The sun generally sets in the west. More precisely, it sets due west only around the equinoxes...
```

### `graph/nodes/arbitrator.py`

This performs deterministic conflict detection for:

- Different years or dates
- Different numeric values
- Boolean contradictions such as available vs discontinued

Detected conflicts are resolved by the Phase 1 `arbitrator_chain`, so local runs use the deterministic fake LLM fallback and live runs can use Gemini through the same chain contract.

It returns:

- `conflicts_detected`
- `arbitration_results`

### `graph/nodes/synthesizer.py`

This calls the Phase 1 `synthesizer_chain` and writes:

- `streaming_answer`
- `final_answer`

It also adds a progressive confidence marker:

- `[HIGH CONFIDENCE]` when usable evidence is strong enough and not partial
- `[LOW CONFIDENCE]` when evidence is missing, failed, low confidence, or partial

`graph/query_mind_graph.py` also exposes `stream_query_graph_events(...)`, which streams LangGraph node events and token-level answer events from `astream_events(...)`. The FastAPI SSE endpoint uses this graph event stream rather than waiting for the final state and splitting it afterward.

Debugging lesson:

> Partial evidence should not produce `[HIGH CONFIDENCE]`. A result with `status="partial"` and `confidence=0.55` can still answer the user, but the marker must reflect uncertainty caused by the extraction timeout.

### `graph/nodes/critic.py`

This calls the Phase 1 `critic_chain`.

If the score is low or the answer contains sensitive-topic flags, the graph pauses with:

```python
interrupt(...)
```

The run can later continue with:

```python
Command(resume={"approved": True})
```

Why this matters:

> Human-in-the-loop review is not bolted onto the API. It is part of the graph control flow and benefits from checkpointed state.

Debugging lesson:

> LangGraph's raw `Interrupt` object is not JSON serializable. The API must never save or return raw `__interrupt__` state. QueryMind projects it into a serializable `interrupts` list and sets `requires_human_review=true`.

Example API-safe interrupt:

```json
{
  "requires_human_review": true,
  "interrupts": [
    {
      "id": "...",
      "value": {
        "reason": "low_confidence_or_sensitive_topic",
        "answer": {
          "answer_text": "[LOW CONFIDENCE] No evidence found.",
          "citations": [],
          "confidence": 0
        }
      }
    }
  ]
}
```

### `graph/checkpointer.py`

This returns a LangGraph checkpointer.

Current behavior:

- `env="dev"` uses async-compatible `InMemorySaver`
- `env="test"` uses `InMemorySaver`
- user-visible session/cache state is persisted by `SessionStore`

Debugging lesson:

> The synchronous SQLite checkpointer from `langgraph-checkpoint-sqlite` does not support async graph methods such as `ainvoke(...)`. FastAPI uses async endpoints, so local dev uses an async-compatible graph checkpointer and leaves durable app-level persistence to `SessionStore`.

### `core/memory/session_store.py`

This is now SQLite-backed.

It stores:

- Latest session state
- Query cache entries
- Query token vectors for lexical cosine similarity fallback
- Query embedding vectors when `GOOGLE_API_KEY` is configured
- Cached retrieval results and final answers

Cache behavior:

- Similarity threshold: `0.92`
- TTL: 1 hour
- Same-session cache lookup before planning
- Embedding cosine similarity is preferred when embeddings are available
- Lexical cosine similarity remains the offline fallback

## FastAPI Integration

`api/server.py` now calls the graph runtime instead of manually running planner/retriever/synthesizer chains.

Important endpoints:

- `POST /query`
- `POST /query/stream`
- `GET /query/{session_id}/state`
- `POST /query/{session_id}/resume`

`/state` can inspect checkpointed graph state, and `/resume` continues a paused human-review run.

Operational notes:

- `/query` returns the projected current-run state, not raw checkpoint internals.
- `/query/{session_id}/state` returns the current projected checkpoint state and `next`.
- If `next` contains `critic`, the graph is paused for review.
- Resume with:

```json
{
  "feedback": {
    "approved": true
  }
}
```

## What Works Today

The current Phase 2 implementation supports:

- Full graph execution
- Dynamic fan-out with `Send`
- Parallel retriever branch merging with reducers
- Stage-specific retriever timeout fallback
- Structured retriever errors with stage/type/timing details
- Current-run state projection with session history preservation
- Conflict detection
- Chain-backed conflict arbitration
- Graph event and token streaming
- Human-in-the-loop interrupt and resume
- SQLite session and embedding-aware cache storage
- Cache-hit routing that skips planning
- FastAPI graph-backed query and resume endpoints
- Local deterministic tests without API keys
- Opt-in live LangSmith integration test

Verification command:

```bash
uv run pytest -q
```

Current result:

```text
31 passed, 1 skipped
```

## Phase 2 Checkpoint Evidence

The checkpoint is covered by `tests/unit/test_graph_phase2.py`.

| Checkpoint item | Status |
|---|---|
| Full graph executes end-to-end on a test query | Verified |
| Parallel retrievers run concurrently | Verified with overlapping trace windows |
| Conflict detection triggers on contradictory dates | Verified |
| Human-in-the-loop interrupt works and resumes | Verified with `update_state(...)` plus `Command(resume=...)` |
| Cache hit and >50% latency drop | Verified; local probe showed about 90% lower latency |
| Simulated retriever timeout returns failed result | Verified |
| LangSmith topology trace | Verified with a root `LangGraph` run and per-node latency entries |

## Production Features Removed From Limitations

The previously conservative Phase 2 items now have concrete implementations:

- Conflict arbitration runs through `arbitrator_chain`; tests monkeypatch that chain contract while live mode can use Gemini.
- Token-level graph event streaming is exposed through `stream_query_graph_events(...)` and used by `/query/stream`.
- The session cache stores and compares embeddings when available, with lexical similarity as the offline fallback.
- LangSmith tracing has an opt-in integration test: run `RUN_LANGSMITH_TESTS=true uv run pytest tests/integration/test_langsmith_tracing.py -q`.

## Debugging Timeline And Fixes

### 1. State Contamination Across Session Runs

Symptom:

```json
"sub_questions": [{"id": "q1"}],
"retrieval_results": [
  {"sub_question_id": "earth_orbit_sun"},
  {"sub_question_id": "sun_set_direction"},
  {"sub_question_id": "q1"}
]
```

Root cause:

> `retrieval_results` and `agent_traces` use reducers so parallel branches merge correctly. But reducers are append-oriented, so when the same `thread_id` is reused, previous run results remain in checkpoint state.

Fix:

- Add a unique `run_id` for every graph invocation.
- Tag retrieval results and traces with `run_id`.
- Return a projected current-run view from the API.
- Move previous completed runs into `session_history`.

Why:

> Session memory should help with follow-ups, but current-run working state must remain clean.

### 2. Raw Interrupt Serialization Crash

Symptom:

```text
TypeError: Object of type Interrupt is not JSON serializable
when serializing dict item '__interrupt__'
```

Root cause:

> The graph paused correctly, but the API tried to persist LangGraph's raw `Interrupt` object.

Fix:

- Strip `__interrupt__` from projected state.
- Convert it into serializable `interrupts`.
- Set `requires_human_review=true`.

Why:

> Human review is expected graph behavior, not an API crash.

### 3. Retriever Timeout Hid Useful Evidence

Symptom:

```json
"answer_text": "No evidence found.",
"error": "retriever timed out after 15s"
```

Root cause:

> Search and extraction were wrapped together. When extraction timed out, any search evidence already found was discarded.

Fix:

- Run search first with its own timeout.
- Run extraction second with its own timeout.
- Preserve search snippets as `status="partial"` if extraction times out.

Why:

> Partial evidence is better than no evidence, but must be marked honestly.

### 4. Generic Timeout Messages

Symptom:

```json
"error": ""
```

Root cause:

> `asyncio.wait_for(...)` raises `TimeoutError()` without a provider message when it cancels the underlying Gemini/LangChain task.

Fix:

- Store structured error records with:
  - `stage`
  - `type`
  - `message`
  - `raw_message`
  - `elapsed_ms`
  - `timeout_seconds`
  - `search_results`
  - `traceback_tail`

Why:

> We cannot invent a Gemini provider error if cancellation happened first, but we can surface exactly what timed out and what evidence was recovered.

### 5. Misleading Confidence Marker

Symptom:

```json
"streaming_answer": "[HIGH CONFIDENCE] ...",
"confidence": 0.55,
"status": "partial"
```

Root cause:

> The synthesizer marked any usable evidence as high confidence, even if that evidence was partial due to timeout.

Fix:

- `[HIGH CONFIDENCE]` only when average evidence confidence is high and no evidence is partial.
- Partial evidence now produces `[LOW CONFIDENCE]`.

Why:

> A useful answer can still be low confidence. The UI marker should match the evidence quality.

### 6. Poor Query Fan-Out For Vague Direction Fragments

Symptom:

```json
"sub_questions": [
  {"question": "where does sun rise from?"},
  {"question": "which direction?"}
]
```

Root cause:

> Simple comma splitting created a vague sub-question that was too weak for retrieval.

Fix:

- Fold vague fragments such as `which direction?` back into the main question.

Why:

> Fan-out should increase recall, not create unsearchable fragments.

### 7. Long Raw Snippet Answers

Symptom:

> Partial fallback answers became long concatenations of search snippets.

Fix:

- Add concise fallback summaries for common sun rise/set evidence.
- Keep citations from the search results.

Why:

> Fallback mode should still produce a readable answer instead of exposing raw retrieval noise.

## Interview Defense

> Phase 2 is where the project becomes a real multi-agent system. I use LangGraph reducers to merge concurrent retriever outputs, `Send` to dynamically fan out sub-questions, checkpointing to preserve state, and `interrupt()` to pause low-confidence answers for human review. The important engineering lesson was separating append-only checkpoint state from current-run state: reducers are great for fan-in, but API responses need a projected view keyed by `run_id`. I also split retrieval into search and extraction stages so timeouts degrade into partial cited answers instead of total failure, with structured errors showing exactly which stage failed and why.

## Next Best Step

Phase 3 should wrap this compiled LangGraph graph inside a Google ADK agent.

The clean integration point is `run_query_graph(query, session_id)`, exposed from `graph/query_mind_graph.py`. ADK can treat the whole graph as one tool while LangGraph remains responsible for stateful orchestration.
