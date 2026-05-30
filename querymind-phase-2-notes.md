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
- `graph_config(session_id)`

Why this matters:

> The API no longer owns orchestration logic. It now calls a compiled graph, which is inspectable, resumable, and checkpointed.

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
- Has a timeout
- Returns a failed retrieval result instead of raising
- Appends its own trace entry

Why this matters:

> One failing retriever should degrade the answer, not crash or hang the graph.

### `graph/nodes/arbitrator.py`

This performs deterministic conflict detection for:

- Different years or dates
- Different numeric values
- Boolean contradictions such as available vs discontinued

It returns `conflicts_detected` and `arbitration_results`.

This is intentionally deterministic in local mode. A live LLM arbitrator can replace the rule-based scorer later without changing the graph contract.

### `graph/nodes/synthesizer.py`

This calls the Phase 1 `synthesizer_chain` and writes:

- `streaming_answer`
- `final_answer`

It also adds a progressive confidence marker:

- `[HIGH CONFIDENCE]` when usable evidence is present
- `[LOW CONFIDENCE]` when evidence is missing, failed, or low confidence

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

### `graph/checkpointer.py`

This returns a LangGraph checkpointer.

Current behavior:

- `env="dev"` uses SQLite via `langgraph-checkpoint-sqlite`
- `env="test"` uses `InMemorySaver`
- If SQLite setup fails, it falls back to `InMemorySaver`

### `core/memory/session_store.py`

This is now SQLite-backed.

It stores:

- Latest session state
- Query cache entries
- Query token vectors for lexical cosine similarity
- Cached retrieval results and final answers

Cache behavior:

- Similarity threshold: `0.92`
- TTL: 1 hour
- Same-session cache lookup before planning

## FastAPI Integration

`api/server.py` now calls the graph runtime instead of manually running planner/retriever/synthesizer chains.

Important endpoints:

- `POST /query`
- `POST /query/stream`
- `GET /query/{session_id}/state`
- `POST /query/{session_id}/resume`

`/state` can inspect checkpointed graph state, and `/resume` continues a paused human-review run.

## What Works Today

The current Phase 2 implementation supports:

- Full graph execution
- Dynamic fan-out with `Send`
- Parallel retriever branch merging with reducers
- Retriever timeout fallback
- Conflict detection
- Human-in-the-loop interrupt and resume
- SQLite session and cache storage
- Cache-hit routing that skips planning
- FastAPI graph-backed query and resume endpoints
- Local deterministic tests without API keys

Verification command:

```bash
uv run pytest -q
```

Current result:

```text
19 passed
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

## Current Limitations

The local Phase 2 graph is production-shaped, but a few pieces are intentionally conservative:

- Conflict arbitration is deterministic instead of live LLM-based in tests.
- Progressive answer streaming is represented as `streaming_answer`; token-level graph event streaming can be expanded further.
- Semantic cache uses lexical cosine similarity locally. Embedding-based cache similarity can be added once the production embedding path is enabled.
- LangSmith tracing is verified with live credentials, but not asserted in unit tests because it depends on external network access.

## Interview Defense

> Phase 2 is where the project becomes a real multi-agent system. I use LangGraph reducers to merge concurrent retriever outputs, `Send` to dynamically fan out sub-questions, checkpointing to preserve state, and `interrupt()` to pause low-confidence answers for human review. The graph survives partial retrieval failures and can skip expensive planning on cache hits.

## Next Best Step

Phase 3 should wrap this compiled LangGraph graph inside a Google ADK agent.

The clean integration point is `run_query_graph(query, session_id)`, exposed from `graph/query_mind_graph.py`. ADK can treat the whole graph as one tool while LangGraph remains responsible for stateful orchestration.
