# QueryMind

Adaptive multi-agent search intelligence system for answering research-style questions with citations, confidence scoring, session memory, and human review checkpoints.

QueryMind is a Python backend that routes simple factual questions to a fast grounded path and sends complex or high-stakes questions through a LangGraph research workflow. The deep workflow plans sub-questions, retrieves evidence, resolves conflicts, synthesizes cited answers, critiques the result, and persists session state for follow-up questions.

## Why I Built This

Search answers often fail in two places: they either respond too quickly without enough evidence, or they overuse expensive multi-agent workflows for simple questions. QueryMind explores a hybrid architecture where query complexity determines the path:

- Fast path for direct factual lookups with Google Search grounding.
- Deep path for comparison, research, current, sensitive, or multi-source questions.
- Human review handoff when evidence is weak, confidence is low, or the topic is sensitive.

The goal is to make AI search feel more like a careful research assistant than a single-shot chatbot.

## Features

- Rule-based ADK router that classifies questions as `fast`, `deep`, or `clarify`.
- LangGraph orchestration with planner, retriever, arbitrator, synthesizer, critic, and cache nodes.
- LangChain LCEL chains with structured Pydantic outputs and deterministic offline fallbacks.
- Tavily-powered web retrieval with retries, concurrency limits, and normalized citations.
- Google Gemini integration for generation, grounding, and optional embeddings.
- SQLite-backed session state and query cache with lexical or embedding similarity.
- Streaming Server-Sent Events endpoint for graph progress and answer tokens.
- Human-in-the-loop resume flow for low-confidence answers.
- Unit and integration tests covering routing, graph execution, caching, streaming, parsing, and tools.

## Architecture

```text
User Query
   |
   v
ADK Router
   |-----------------------------|
   |                             |
   v                             v
Fast Grounding              QueryMind Deep Graph
Google Search               initialize
   |                         cache_lookup
   v                         planner
Cited Answer                retriever(s)
                             arbitrator
                             synthesizer
                             critic
                             cache_store / human review
```

## Tech Stack

- Python 3.10+
- FastAPI and Uvicorn
- LangGraph
- LangChain / LCEL
- Google ADK and Gemini
- Tavily Search
- Pydantic
- SQLite
- Pytest and pytest-asyncio

## Project Structure

```text
api/                  FastAPI server and HTTP endpoints
adk/                  Query router, grounding boundary, ADK root agent
agents/               Phase 1 agent node implementations
core/
  chains/             Prompts, parsers, and LCEL chains
  memory/             SQLite session state and semantic query cache
  tools/              Web search and document store tools
graph/                LangGraph state, nodes, edges, and checkpointer setup
observability/        Tracing/dashboard scaffolding
tests/                Unit and integration tests
```

## Getting Started

Run commands from the repository root.

```bash
uv sync
```

Create a `.env` file if you want live model and search calls:

```bash
GOOGLE_API_KEY=your_google_or_gemini_key
TAVILY_API_KEY=your_tavily_key

# Optional
QUERYMIND_GROUNDING_MODEL=gemini-2.5-flash
QUERYMIND_SESSION_DB=.querymind_sessions.sqlite3
QUERYMIND_CHECKPOINT_DB=.querymind_checkpoints.sqlite3
TAVILY_CONCURRENCY=5
```

For fully local deterministic runs, omit API keys or set:

```bash
QUERYMIND_FORCE_FAKE_LLM=true
```

## Run the API

```bash
uv run python -m uvicorn api.server:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Deep Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare Python 2 vs Python 3 release dates and cite sources"}'
```

### Routed Query

```bash
curl -X POST http://127.0.0.1:8000/query/orchestrated \
  -H "Content-Type: application/json" \
  -d '{"query":"Where does the sun rise?"}'
```

### Stream Events

```bash
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Research RAG vs long-context tradeoffs and cite evidence"}'
```

### Inspect or Resume a Session

```bash
curl http://127.0.0.1:8000/query/<session_id>/state

curl -X POST http://127.0.0.1:8000/query/<session_id>/resume \
  -H "Content-Type: application/json" \
  -d '{"feedback":{"approved":true}}'
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/query` | Run the LangGraph deep research workflow. |
| `POST` | `/query/orchestrated` | Route to fast grounding or deep research based on query complexity. |
| `POST` | `/query/stream` | Stream graph node events and answer tokens with SSE. |
| `GET` | `/query/{session_id}/state` | Inspect persisted state and graph checkpoint data. |
| `POST` | `/query/{session_id}/resume` | Resume a human-review-interrupted run. |

## Testing

```bash
uv run python -m pytest
```

The default test configuration sets `QUERYMIND_FORCE_FAKE_LLM=true`, so the suite can run without external API keys.

Live LangSmith tracing checks are opt-in:

```bash
RUN_LANGSMITH_TESTS=true LANGSMITH_API_KEY=your_key uv run python -m pytest tests/integration
```

## Implementation Highlights

- **Complexity-aware routing:** direct questions avoid the full graph, while comparison, current, recommendation, sensitive, and evidence-seeking queries use the deep pipeline.
- **Structured outputs:** planner, retriever, synthesizer, critic, and arbitrator outputs are parsed into Pydantic models with JSON repair fallback.
- **Parallel retrieval:** sub-questions fan out concurrently, then merge into a shared graph state.
- **Conflict handling:** arbitration logic detects contradictory dates, low-confidence evidence, and source quality differences.
- **Reviewable answers:** the critic can interrupt execution and require explicit resume feedback before caching the final answer.
- **Session continuity:** follow-up questions can be rewritten using prior session context, and similar queries can reuse cached results.

## Current Status

QueryMind is a backend-first prototype with a tested orchestration layer and API surface. The observability dashboard is currently scaffolding; tracing hooks and integration-test coverage are in place for future expansion.

## Roadmap

- Build a polished Streamlit or web dashboard for graph traces, confidence scores, and session history.
- Add richer source-ranking heuristics and citation quality scoring.
- Expand document ingestion beyond web search into uploaded files and private knowledge bases.
- Add deployment templates for containerized API hosting.
- Improve evaluation with benchmark query sets and regression scoring.
