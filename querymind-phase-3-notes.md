# QueryMind Phase 3 Notes

## Current Project Stage

QueryMind is now in the **Phase 3 Google ADK orchestration** stage.

Phase 1 built reliable LangChain tools and LCEL chains. Phase 2 turned those pieces into a stateful LangGraph multi-agent pipeline. Phase 3 adds the top-level orchestration layer that decides whether a user query should use:

- A fast grounded answer path
- The full QueryMind LangGraph research pipeline
- A clarification path for ambiguous follow-ups

The main idea of this phase is not "more agents." It is **routing**.

Simple factual questions should not pay the cost of planner -> retrievers -> arbitrator -> synthesizer -> critic. Complex, sensitive, comparative, conflicting, or current-information questions still go through the full graph.

## Current Flow

```text
User query
   |
   v
/query/orchestrated
   |
   v
load prior session state
   |
   v
resolve small follow-up query if possible
   |
   v
deterministic pre-router
   |
   |-- fast --> Gemini Google Search grounding
   |
   |-- deep --> QueryMind LangGraph pipeline
   |
   |-- clarify --> ask for clarification
   |
   v
normalized API response with route_decision, citations, confidence, errors, and state
```

## Why ADK Now

Google ADK is introduced after LangGraph because it belongs at the conversation and orchestration boundary.

LangGraph still owns:

- Stateful multi-agent research
- Parallel fan-out and fan-in
- Human-in-the-loop review
- Cache and graph execution state

ADK now owns:

- The root agent shell
- Google Search grounding integration
- Simple-vs-complex routing
- Tool exposure through `FunctionTool`
- Callback hooks for later observability
- Session and multi-turn handoff behavior

Interview defense:

> I used LangGraph for the heavy stateful research graph and ADK as the top-level routing/conversation layer. This mirrors a production search assistant architecture: route cheap factual requests to a fast grounded path, and send only complex requests into the expensive multi-agent backend.

## Important Files

### `adk/router.py`

This is the deterministic, non-LLM pre-router.

It returns a `RouteDecision` with:

- `route`: `fast`, `deep`, or `clarify`
- `reason`
- `complexity_score`
- `matched_rules`

Fast path rules cover short factual queries such as:

- `where is ...`
- `when did ...`
- `who built ...`
- `what is ...`
- definitions
- capital-city questions
- simple sun rise/set direction questions

Deep path rules cover:

- comparisons
- research/deep-dive requests
- citation-heavy prompts
- latest/current/today questions
- recommendations
- conflict/fact-checking
- medical/legal/financial/investment topics
- compound queries

Why this matters:

> The fastest LLM call is the one you never make. A deterministic router prevents simple questions from accidentally entering the full graph and burning planner/retriever/synthesizer calls.

### `adk/orchestrator.py`

This is the Phase 3 orchestration layer.

It exposes:

- `route_query(...)`
- `build_root_agent(...)`
- `resolve_follow_up_query(...)`

`route_query(...)` does the actual runtime dispatch:

```text
RouteDecision.FAST    -> answer_with_grounding(...)
RouteDecision.DEEP    -> run_querymind(...)
RouteDecision.CLARIFY -> clarification response
```

It also records route events through `record_event(...)` so Phase 4 can show route decisions and latency in the dashboard.

Structured errors are surfaced here. If the fast grounding path or deep QueryMind provider throws, the API gets:

```json
{
  "error": {
    "stage": "fast_grounding",
    "type": "RuntimeError",
    "message": "grounding boom"
  }
}
```

Debugging lesson:

> Route/tool failures should not collapse into a generic 500 or vague answer. They should become typed, stage-specific error objects so the frontend and dashboard can show exactly what failed.

### `adk/tools/querymind_tool.py`

This wraps the existing LangGraph runtime as an ADK `FunctionTool`.

The callable is:

```python
async def run_querymind(query: str, session_id: str) -> dict[str, Any]:
    ...
```

It returns:

- `answer`
- `citations`
- `confidence`
- `requires_human_review`
- `state`

Why this matters:

> QueryMind's full graph becomes one tool from ADK's point of view. ADK does not need to know planner/retriever/arbitrator internals; it only needs a clean tool contract.

### `adk/grounding.py`

This implements the fast Google Search grounding boundary.

When live credentials are available and fake mode is off, it calls Gemini with the Google Search tool:

```python
types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.0,
)
```

It parses grounding metadata into QueryMind citation dictionaries:

```json
{
  "source": "...",
  "url": "...",
  "excerpt": ""
}
```

Local tests still use deterministic offline grounding because `QUERYMIND_FORCE_FAKE_LLM=true` is set in the test environment.

Why this matters:

> The production path can use live Google Search grounding, while tests remain stable, fast, and network-free.

### `adk/session_manager.py`

This provides a small local session manager shape for Phase 3 tests and future ADK session integration.

It stores:

- `user_id`
- `session_id`
- session state

The FastAPI endpoint currently loads durable state from `SessionStore` and passes it to `route_query(...)`.

### `adk/callbacks.py`

This records ADK lifecycle events in memory.

It supports:

- `before_model_callback`
- `after_model_callback`
- `on_model_error_callback`
- `record_event(...)`
- `get_events()`
- `clear_events()`

Current event examples:

- `route_decision`
- `route_complete`
- `route_error`
- `before_model`
- `after_model`
- `model_error`

Why this matters:

> Phase 4's observability dashboard needs a real event stream. These callbacks establish the data contract before building the UI.

### `api/server.py`

Phase 3 adds:

```text
POST /query/orchestrated
```

This endpoint is different from:

```text
POST /query
```

`/query` always runs the full LangGraph pipeline.

`/query/orchestrated` routes first:

```text
simple factual query -> fast grounding
complex query        -> full QueryMind graph
ambiguous follow-up  -> clarification
```

The orchestrated response includes:

- `session_id`
- `route`
- `answer`
- `citations`
- `confidence`
- `requires_human_review`
- `route_decision`
- `state`
- `error`

### `graph/nodes/arbitrator.py`

Phase 3 also upgrades the arbitrator so the resume claim about source-aware arbitration is more truthful.

Conflict arbitration now includes component scoring:

- `authority`
- `recency`
- `corroboration`
- `total`

Authority scoring gives higher weight to first-party or official sources such as:

- `.gov`
- `.edu`
- `python.org`
- `docs.*`
- `cloud.google.com`
- `ai.google.dev`

Lower authority is assigned to community/social sources such as:

- Reddit
- Quora
- Facebook

Why this matters:

> The arbitrator is no longer only asking an LLM to pick a winner. It now attaches deterministic source-quality signals that can be inspected, tested, and shown in the dashboard.

## Multi-Turn Follow-Up Handling

Phase 3 adds a small follow-up rewrite step.

Example:

```text
Previous query: Tell me about LangGraph
New query:      Who built it?
Routed query:   Who built LangGraph?
```

This is intentionally conservative. It only rewrites short pronoun follow-ups when a prior topic can be inferred from session state.

Debugging lesson:

> Follow-up resolution should happen before routing. Otherwise "Who built it?" may be classified as ambiguous even when the session already contains enough context.

## Google Search Grounding Behavior

Google Search grounding is not the same as Tavily search.

With Tavily, QueryMind receives search result objects directly.

With Gemini Google Search grounding, Gemini receives the search tool and returns:

- answer text
- grounding metadata
- grounding chunks
- web search queries
- citation/support information

QueryMind converts the grounding chunks into its standard citation format.

Current behavior:

- Live mode: uses `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Test/fake mode: uses deterministic offline answers
- If live grounding fails, the answer falls back to offline grounding and includes `live_grounding_error` metadata

## What Works Today

Phase 3 currently has:

- ADK `FunctionTool` wrapper for the full QueryMind graph
- ADK root agent builder with Google Search and QueryMind tools
- Deterministic pre-router with 10+ route tests
- Fast path for simple factual queries
- Deep path for complex/sensitive/comparative/current queries
- Google Search grounding call path using Gemini
- Grounding metadata to citation parsing
- Multi-turn follow-up rewrite for simple pronoun references
- Structured route/tool errors
- ADK callback event collection
- Source-aware arbitration scoring
- `/query/orchestrated` API endpoint

Verification command:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

Latest verified result:

```text
51 passed, 1 skipped
```

The skipped test is the opt-in live LangSmith integration test.

## Phase 3 Checkpoint Evidence

| Checkpoint item | Status |
|---|---|
| ADK root agent correctly routes simple vs complex queries | Verified with 10+ route tests |
| Deterministic pre-router sends simple factual queries to fast path | Verified |
| Complex/sensitive/comparative/current queries route to QueryMind | Verified |
| Google Search grounding adds citations to final answer | Code path implemented; metadata parsing tested offline |
| Multi-turn conversation follow-up resolves correctly | Verified with `"Who built it?"` rewrite test |
| ADK callbacks feed observability dashboard with real data | Event collection implemented and tested |
| Route/grounding/tool errors surface as structured errors | Verified |
| Arbitration scores authority, recency, and corroboration | Verified |
| Complex query latency under 12 seconds | Not yet verified |

## Current Limitations

- The full ADK Runner conversation loop is not yet exposed as the main local runtime; `/query/orchestrated` is currently the practical Phase 3 entrypoint.
- Live Google Search grounding requires network access and valid Gemini credentials. Unit tests intentionally use offline grounding.
- Grounding metadata parsing currently extracts source title and URL; richer citation support ranges can be added later.
- Multi-turn follow-up rewriting is intentionally conservative and does not yet perform full conversational coreference resolution.
- The source-aware arbitration scores are heuristic. They are useful and testable, but not a complete trust-and-safety ranking system.
- The `< 12s` complex-query latency target is not yet proven. The graph can still be slow when live Gemini/Tavily calls are involved.

## Debugging Lessons From Phase 3

### 1. Routing Must Happen Before the Expensive Graph

The full QueryMind graph is powerful but expensive. Simple factual questions do not need planning, fan-out, arbitration, and critic review.

The deterministic pre-router gives predictable latency and cost control.

### 2. Tests Should Not Require Google Search

Grounding is a live external dependency. Tests stay offline through the fake LLM mode and dependency-injectable providers.

This keeps the suite reliable while preserving the real production path.

### 3. Grounding Metadata Is a Different Shape Than Search Results

Google Search grounding returns model metadata, not a Tavily-style list of results.

So Phase 3 added a parser that converts grounding chunks into QueryMind's citation format.

### 4. Errors Need Stage Names

"Something failed" is not enough once there are multiple paths.

Phase 3 errors include:

- route
- stage
- exception type
- message

This will make Phase 4 observability much easier.

### 5. Resume Claims Need Implementation Behind Them

The roadmap/resume language mentioned source authority, recency, and corroboration. Phase 3 added deterministic component scores so that claim is backed by code and tests.

## Good Demo Queries

Fast path:

```text
Where does the sun rise?
```

Deep path:

```text
Compare Python 2 vs Python 3 release dates and resolve conflicting sources.
```

Sensitive/deep path:

```text
Should I invest in Tesla stock?
```

Follow-up context:

```text
Tell me about LangGraph.
Who built it?
```

## Next Best Phase

The next phase should be **Phase 4: Observability Dashboard**.

Phase 3 now records the right data:

- route decisions
- selected path
- route errors
- grounding chunk counts
- confidence
- arbitration scores
- graph traces

Phase 4 should turn those into a live dashboard that proves the architecture visually.

