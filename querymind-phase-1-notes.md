# QueryMind Phase 1 Notes

## Current Project Stage

QueryMind is currently in the **Phase 1 LangChain foundation** stage.

At this point, the project is not yet a full LangGraph multi-agent system. The goal of this phase is to build the reusable pieces that the future graph will coordinate:

- Web search tool
- Document store and semantic search layer
- Structured output parsers
- Prompt templates
- LCEL chains for planning, retrieval, synthesis, and critique
- Basic FastAPI streaming endpoint

The core idea is simple: before building complex agent orchestration, first build reliable, typed, testable tools and chains.

## Current Flow

```text
User query
   |
   v
planner_chain
   |
   v
sub-questions
   |
   v
retriever_chain for each sub-question
   |
   v
retrieved evidence
   |
   v
synthesizer_chain
   |
   v
final cited answer
   |
   v
critic_chain can evaluate confidence
```

The API currently uses a simple async fan-out with `asyncio.create_task`. Full dynamic fan-out, fan-in, checkpointing, and stateful orchestration will come later in the LangGraph phase.

## Why LangChain First

LangChain is used first because it is best suited for the **tool and chain layer**.

LangChain gives the project:

- Tool wrappers such as `StructuredTool`
- Prompt templates
- Output parsers
- LCEL composition with the `|` operator
- Standard `.invoke()`, `.ainvoke()`, `.stream()`, and `.astream()` interfaces
- Model swapping and fallback patterns

Interview defense:

> I started with LangChain because I wanted each reasoning unit to be independently testable before adding graph orchestration. LangChain gives me a clean contract for tools, prompts, parsers, and chains. Once those units are stable, LangGraph can coordinate them without mixing orchestration logic with model and tool logic.

## Important Files

### `core/tools/web_search.py`

This is the external retrieval tool.

It uses Tavily's async client when `TAVILY_API_KEY` is configured. It also includes:

- `SearchResult` Pydantic model
- Retry logic with `tenacity`
- Concurrency limiting with `asyncio.Semaphore`
- LangChain `StructuredTool` wrapper

Why this matters:

> Search is an unreliable external dependency, so I wrapped it with typed outputs, retry logic, and concurrency control. That makes it safer to call under future parallel fan-out.

### `core/tools/document_store.py`

This stores and searches evidence chunks.

It can use Gemini embeddings plus FAISS when configured, but falls back to lexical ranking locally.

Why this matters:

> I wanted the project to remain runnable without paid API keys, while still having a production path. So the same interface works offline with lexical search and online with embeddings plus vector search.

### `core/chains/parsers.py`

This defines the structured schemas passed between chains:

- `SubQuestion`
- `SubQuestionList`
- `Citation`
- `CitedAnswer`
- `RetrievalEvidence`
- `ConfidenceScore`

Why this matters:

> I avoid free-form LLM text between stages. Each chain returns a Pydantic-validated object, which makes later orchestration safer because every node has predictable input and output.

### `core/chains/prompts.py`

This defines role-specific prompts:

- Planner prompt
- Retriever prompt
- Synthesizer prompt
- Critic prompt

Each prompt includes parser format instructions.

Why this matters:

> The parser schema is injected into the prompt, so the model is explicitly guided to return machine-readable JSON.

### `core/chains/base_chains.py`

This wires prompts, models, parsers, and fallback logic into LCEL chains.

The general shape is:

```python
planner_chain = (
    PLANNER_PROMPT
    | model
    | StrOutputParser()
    | SubQuestionParser
)
```

Why this matters:

> LCEL lets me compose each chain declaratively. The chain is not just a function; it is a runnable object that supports async invocation, streaming, batching, tracing, and later integration into LangGraph.

### `api/server.py`

This exposes a basic FastAPI streaming endpoint.

Current responsibility:

- Accept a user query
- Run planner, retriever, and synthesizer chains
- Stream the final answer token by token over SSE
- Store simple session state

This is intentionally simple for Phase 1. It is not the final orchestration layer.

## Why Fallbacks Exist

The project supports two modes:

1. **Local learning mode**: no API keys required
2. **Real model mode**: Gemini and Tavily are used when environment keys exist

This is deliberate.

Interview defense:

> I built deterministic fallbacks so tests and development are stable without network calls. But the architecture still uses real LangChain interfaces, so switching to production APIs is configuration-driven, not a rewrite.

## What Works Today

The current Phase 1 system has:

- A working LangChain foundation
- Async web-search tool contract
- Structured Pydantic outputs
- LCEL chains
- Offline testability
- Basic FastAPI streaming
- Unit tests proving core behavior

Verification command:

```bash
uv run pytest -q
```

Expected result:

```text
9 passed
```

## Current Live Latency Observations

When real API keys are configured, QueryMind stops using the local fake fallback path and starts calling real external services.

That means live responses can be much slower than unit tests.

One observed query:

```text
where is england located and how to address the accent gaps? where is the best english learning classes found?
```

produced this timing:

```text
total_latency_ms: 85384
```

So the full response took about **85 seconds**.

The trace showed:

```text
planner:      ~38.3s
synthesizer:  ~40.6s
```

This means most of the time was spent in real Gemini model calls, not FastAPI.

## Why It Takes Time Right Now

The current live pipeline is:

```text
1 planner LLM call
3 Tavily/search calls
3 retriever LLM calls
1 synthesizer LLM call
```

The exact number depends on how many sub-questions the planner creates.

For a compound query, the planner may split the original question into several sub-questions. In the observed example, it created:

- England location
- Accent gap strategies
- Best English learning classes

That is good behavior from a search quality perspective, but it increases latency.

The main reasons for current slowness are:

- Real Gemini calls are slower than the local fake fallback.
- Planner and synthesizer currently use Pro-class models.
- Swagger waits for the full JSON response before showing anything.
- The current phase prioritizes correctness, structure, and grounded retrieval over latency.
- The system has not yet added caching, timeouts, progressive streaming, or smarter model routing.

## Will LangGraph Make This Faster?

LangGraph will not magically make an individual LLM call faster.

What LangGraph improves is **orchestration**:

- Cleaner parallel fan-out and fan-in
- Better state tracking
- Conditional routing
- Timeouts and fallback paths
- Checkpointing and resume
- Streaming intermediate progress
- Skipping unnecessary nodes
- Cleaner observability for each agent/node

So LangGraph mainly improves:

- perceived responsiveness
- reliability
- debuggability
- control over multi-step execution

Actual latency improvements will come from:

- using faster models for simpler steps
- caching search and retrieval results
- avoiding unnecessary LLM calls
- streaming partial results
- setting model timeouts
- falling back from Pro to Flash when needed

## Near-Term Performance Improvements

Likely optimizations for later:

1. Use `gemini-2.5-flash` for planner.
2. Use `gemini-2.5-flash` for the first synthesizer pass.
3. Reserve Pro models for complex, ambiguous, or high-stakes questions.
4. Skip retriever LLM summarization for simple searches and pass raw snippets directly to synthesis.
5. Cache repeated sub-query search results.
6. Add timeouts around external API calls.
7. Make `/query/stream` the main user experience instead of relying on Swagger JSON responses.
8. Use LangGraph to stream progress events such as planning, searching, retrieving, and synthesizing.

Interview defense:

> The current latency is expected because one user query expands into multiple LLM and search calls. In this phase, I prioritized correctness, structure, typed outputs, and grounded retrieval over latency. The trace shows most time is spent in planner and synthesizer Gemini Pro calls. In the next phase, LangGraph will improve orchestration through parallel fan-out, checkpointing, conditional routing, and streaming progress. Separately, actual latency can be reduced by using Flash for planner and synthesis, caching sub-query results, and adding timeouts and fallbacks.

## What Is Not Built Yet

These are intentionally left for later phases:

- LangGraph `StateGraph`
- Dynamic `Send` fan-out
- Graph checkpointing
- Arbitration node
- Real progressive streaming from partial retriever results
- Human-in-the-loop interrupt
- Observability dashboard
- Google ADK root agent

Interview defense:

> At this stage, I have completed the reusable LangChain layer. The next phase is LangGraph orchestration, where these chains become graph nodes and the system gains dynamic fan-out, fan-in, checkpointing, and conflict arbitration.

## Architecture Rationale

The architecture is layered:

```text
LangChain = skills, tools, chains
LangGraph = orchestration, state, parallelism
Google ADK = top-level product agent/runtime
FastAPI = serving interface
```

The main design strength is separation of concerns.

Interview defense:

> I did not want graph nodes to contain raw prompt strings, parsing logic, and API details. Instead, each node will call a prebuilt chain or tool. That keeps LangGraph focused on state transitions and failure handling.

## Likely Interview Questions

### Why not jump directly to agents?

Because agents are harder to test when tools and outputs are not stable. Building tools and chains first gives clean contracts.

### Why Pydantic?

Because LLM outputs are unreliable. Pydantic validation turns fuzzy text into typed application data.

### Why Tavily?

Because the system needs web-grounded answers. Tavily provides search results optimized for LLM retrieval workflows.

### Why async?

Because search fan-out is naturally parallel. Async lets multiple retrievers run concurrently without blocking.

### Why LangGraph later?

Because once the individual chains work, LangGraph can coordinate them with shared state, conditional routing, retries, checkpointing, and parallel branches.

### Why fallback local logic?

Because tests should not depend on API keys, network availability, or model randomness.

## One-Line Resume Defense

> QueryMind is a layered multi-agent search system. I first built the LangChain foundation: typed tools, structured parsers, LCEL chains, and async retrieval. This gives every future LangGraph node a stable contract, making the later fan-out, synthesis, and arbitration orchestration much easier to reason about and test.
