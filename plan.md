# 🧠 QueryMind — Adaptive Multi-Agent Search Intelligence System
### A Complete Learning Roadmap: LangChain → LangGraph → Google ADK

> **Resume tagline:** *"Built a production-grade multi-agent search system modelling Google AI Mode — featuring adaptive query fan-out, real-time streaming synthesis, agent disagreement arbitration, persistent memory, and a live observability dashboard."*

---

## 📌 Project Overview

QueryMind is a system that takes any natural language query and:
1. **Decomposes** it into parallel sub-questions (adaptive fan-out)
2. **Dispatches** specialized retriever agents concurrently
3. **Streams** a grounded answer progressively as agents finish
4. **Arbitrates** when agents return conflicting facts
5. **Remembers** past sessions and reuses cached results intelligently
6. **Exposes** a live observability dashboard showing every agent's state, latency, and cost

This is not a tutorial project. Every phase builds real, working code that feeds into the next phase. By the end, you have a single deployable system that demonstrates mastery of three frameworks.

---

## 🗂️ Repository Structure (target end state)

```
querymind/
├── core/
│   ├── tools/              # LangChain tools (Phase 1)
│   ├── chains/             # LangChain chains & parsers (Phase 1)
│   └── memory/             # LangChain memory abstractions (Phase 1)
├── graph/
│   ├── nodes/              # LangGraph node functions (Phase 2)
│   ├── edges/              # Conditional routing logic (Phase 2)
│   ├── state.py            # Shared StateGraph schema (Phase 2)
│   └── checkpointer.py     # Persistence layer (Phase 2)
├── agents/
│   ├── planner.py          # Adaptive query decomposition agent
│   ├── retriever.py        # Parallel web/doc retriever agents
│   ├── synthesizer.py      # Streaming answer synthesis agent
│   ├── arbitrator.py       # Disagreement resolution agent
│   └── critic.py           # Hallucination / confidence checker
├── adk/
│   ├── orchestrator.py     # Google ADK root agent (Phase 3)
│   ├── grounding.py        # Google Search grounding integration
│   └── callbacks.py        # ADK lifecycle hooks
├── observability/
│   ├── tracer.py           # Custom LangSmith + OpenTelemetry tracer
│   └── dashboard/          # Streamlit or FastAPI dashboard
├── api/
│   └── server.py           # FastAPI streaming endpoint
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| LLM backbone | Google Gemini 1.5 Pro / Flash (via LangChain + ADK) |
| Tool layer | LangChain (tools, retrievers, output parsers) |
| Agent orchestration | LangGraph (StateGraph, parallel branches, checkpointing) |
| Top-level orchestration | Google ADK (Agent Development Kit) |
| Web retrieval | Tavily API / SerpAPI + BeautifulSoup |
| Persistence | LangGraph + SQLite (dev) → PostgreSQL (prod) |
| Streaming | Server-Sent Events via FastAPI |
| Observability | LangSmith + OpenTelemetry + custom dashboard |
| UI | Streamlit (for observability dashboard) |
| Testing | pytest + pytest-asyncio |
| Package management | `uv` (fast, modern Python package manager) |

---

## 🗓️ Phase 0 — Environment Setup (Day 1)

**Goal:** Get a clean, reproducible dev environment ready in VS Code.

### Steps

```bash
# 1. Create project
mkdir querymind && cd querymind
uv init
uv add langchain langchain-google-genai langgraph google-adk
uv add langchain-community tavily-python fastapi uvicorn streamlit
uv add langsmith opentelemetry-sdk pytest pytest-asyncio python-dotenv
```

### .env.example
```
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=querymind
LANGCHAIN_TRACING_V2=true
```

### VS Code Extensions to install
- Python (Microsoft)
- Pylance
- Ruff (linter/formatter)
- REST Client (for testing your API endpoints inline)
- GitLens

### Coding Agent Prompt (paste this to Cursor/Copilot/Cline to bootstrap)
```
I'm building a project called QueryMind. It's a multi-agent search system.
Project structure: [paste the tree above]
Tech stack: LangChain, LangGraph, Google ADK, FastAPI, Streamlit.
Always use async/await. Always type-hint. Use Pydantic models for all data structures.
Never use deprecated LangChain v0.1 APIs — use langchain-core and LCEL (LangChain Expression Language).
```

### ✅ Checkpoint
- [ ] `uv run python -c "import langchain, langgraph; print('OK')"` succeeds
- [ ] `.env` loaded, `GOOGLE_API_KEY` accessible
- [ ] LangSmith project visible at smith.langchain.com

---

## 📚 Phase 1 — LangChain: The Tool Layer (Days 2–6)

**Goal:** Build every reusable tool, retriever, and parser that your agents will later call. LangChain is the *foundation* — you're not building agents yet, just the arsenal.

**Core concepts you will learn:**
- LCEL (LangChain Expression Language) — pipe syntax, Runnables
- Custom Tools with `@tool` decorator and `StructuredTool`
- Output parsers (Pydantic, JSON, XML)
- Document loaders and text splitters
- Embeddings and vector stores (FAISS)
- Prompt templates (ChatPromptTemplate, FewShotPromptTemplate)
- Callbacks and streaming handlers

---

### Step 1.1 — Web Retrieval Tool

**File:** `core/tools/web_search.py`

**What to build:** A `@tool`-decorated async function that takes a query string, calls Tavily, returns a list of `SearchResult` Pydantic objects with `url`, `title`, `content`, `score`.

**Key concepts:** `@tool`, `StructuredTool.from_function`, `async def`, Pydantic `BaseModel`

**Coding agent prompt:**
```
Create core/tools/web_search.py.
- Use Tavily async client
- Return List[SearchResult] where SearchResult has url, title, content, relevance_score fields
- Use @tool decorator with a clear docstring (LangChain uses docstrings as tool descriptions)
- Handle rate limits with exponential backoff using tenacity
- Write a pytest test in tests/unit/test_web_search.py
```

**Production challenge you'll hit:** Tavily rate limits under parallel load. Fix: implement a semaphore-based concurrency limiter.

---

### Step 1.2 — Document Chunking & Embedding Tool

**File:** `core/tools/document_store.py`

**What to build:** Takes raw text from search results, chunks it, embeds it with Google's embedding model, stores in FAISS. Exposes a `semantic_search(query, top_k)` tool.

**Key concepts:** `RecursiveCharacterTextSplitter`, `GoogleGenerativeAIEmbeddings`, `FAISS.from_documents`, retriever interface

**Coding agent prompt:**
```
Create core/tools/document_store.py.
- Accept a list of SearchResult objects, chunk and embed them
- Use GoogleGenerativeAIEmbeddings (model: models/embedding-001)
- Store in FAISS in-memory index
- Expose as a LangChain tool: semantic_search(query: str, top_k: int = 5) -> List[str]
- The tool should be callable from a LangChain agent's tool list
```

**Production challenge you'll hit:** FAISS is not thread-safe. Fix: wrap with asyncio.Lock for concurrent access.

---

### Step 1.3 — Output Parsers & Structured Outputs

**File:** `core/chains/parsers.py`

**What to build:** Three parsers your agents will use later:
1. `SubQuestionParser` — parses LLM output into `List[SubQuestion]`
2. `CitedAnswerParser` — parses synthesized answer with inline citation markers
3. `ConfidenceParser` — parses a confidence score + reasoning from critic output

**Key concepts:** `PydanticOutputParser`, `JsonOutputParser`, `OutputFixingParser` (auto-retry on parse failure)

**Coding agent prompt:**
```
Create core/chains/parsers.py.
Define three Pydantic models: SubQuestion(id, question, search_query, reasoning),
CitedAnswer(answer_text, citations: List[Citation], confidence: float),
ConfidenceScore(score: float, reasoning: str, flags: List[str])
Wrap each in a PydanticOutputParser.
Also create an OutputFixingParser wrapper for each that auto-retries with Gemini Flash on parse failure.
```

**Production challenge you'll hit:** LLMs sometimes return malformed JSON. `OutputFixingParser` is your first taste of self-healing chains.

---

### Step 1.4 — Prompt Templates

**File:** `core/chains/prompts.py`

**What to build:** All system + user prompts for every agent role as `ChatPromptTemplate` objects with `partial_variables` for reusable context injection.

**Agent prompts to write:**
- `PLANNER_PROMPT` — decomposes query, reasons about complexity, outputs SubQuestion list
- `RETRIEVER_PROMPT` — given a sub-question and docs, extracts relevant evidence
- `SYNTHESIZER_PROMPT` — combines all evidence into a cited answer, streams token by token
- `ARBITRATOR_PROMPT` — given two conflicting claims + sources, reasons which is more credible
- `CRITIC_PROMPT` — checks final answer for hallucinations, assigns confidence flags

**Key concepts:** `ChatPromptTemplate.from_messages`, `SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`, `partial()`

---

### Step 1.5 — LCEL Chains

**File:** `core/chains/base_chains.py`

**What to build:** Wire prompts + models + parsers into LCEL chains using the `|` pipe syntax.

```python
# Example of what you'll build
planner_chain = PLANNER_PROMPT | gemini_pro | SubQuestionParser()
retriever_chain = RETRIEVER_PROMPT | gemini_flash | CitedAnswerParser()
```

**Key concepts:** `RunnablePassthrough`, `RunnableParallel`, `RunnableLambda`, `.stream()`, `.astream()`, `.batch()`, `.abatch()`

**Coding agent prompt:**
```
Create core/chains/base_chains.py.
Wire up LCEL chains for planner, retriever, synthesizer, arbitrator, critic.
Use ChatGoogleGenerativeAI — gemini-1.5-pro for planner/synthesizer/arbitrator, gemini-1.5-flash for retriever/critic.
All chains must support .astream() for token streaming.
Add a RunnableWithFallbacks wrapper that falls back to gemini-flash if pro times out.
```

**Production challenge you'll hit:** `RunnableWithFallbacks` — how to handle partial failures gracefully without crashing the whole chain.

---

### ✅ Phase 1 Checkpoint
- [ ] All tools pass unit tests
- [ ] FAISS concurrent write issue identified and fixed with asyncio.Lock
- [ ] All 5 LCEL chains stream tokens correctly
- [ ] LangSmith shows chain traces with correct input/output at each step
- [ ] OutputFixingParser successfully auto-heals one malformed JSON response (write a test for this)

---

## 🔀 Phase 2 — LangGraph: The Orchestration Layer (Days 7–16)

**Goal:** Build the actual multi-agent system using LangGraph's `StateGraph`. This is the hardest and most important phase. Every challenge in the project lives here.

**Core concepts you will learn:**
- `StateGraph` and `TypedDict` state schema
- Node functions and edge routing
- `START` / `END` nodes
- Conditional edges and branching
- **Parallel node execution with fan-out/fan-in**
- `Checkpointer` for persistence and resumability
- Human-in-the-loop interrupt/resume
- Streaming events from a running graph
- Subgraphs for modularity

---

### Step 2.1 — Define the Global State

**File:** `graph/state.py`

**What to build:** The `QueryMindState` TypedDict that flows through every node.

```python
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages

class QueryMindState(TypedDict):
    # Input
    original_query: str
    session_id: str

    # Planner outputs
    sub_questions: List[SubQuestion]
    complexity_score: float         # 0-1, determines fan-out count

    # Retriever outputs (one per sub-question)
    retrieval_results: Annotated[List[RetrievalResult], operator.add]  # reducers!

    # Arbitration
    conflicts_detected: List[Conflict]
    arbitration_results: List[ArbitrationResult]

    # Synthesis
    streaming_answer: str
    final_answer: CitedAnswer

    # Critic
    confidence_score: ConfidenceScore
    requires_human_review: bool

    # Memory
    cache_hits: List[str]
    session_history: List[dict]

    # Observability
    agent_traces: Annotated[List[AgentTrace], operator.add]
    total_tokens_used: int
    total_latency_ms: int
```

**Key concept to understand deeply:** `Annotated[List, operator.add]` is a LangGraph **reducer**. Without it, parallel nodes overwrite each other's state. With it, their outputs are merged. This is the key to fan-out/fan-in.

---

### Step 2.2 — Planner Node (Adaptive Fan-out)

**File:** `graph/nodes/planner.py`

**What to build:** A node that calls `planner_chain`, gets `List[SubQuestion]`, and sets `complexity_score` to determine how many retrievers to spawn.

**The adaptive logic:**
```python
def compute_complexity(query: str, sub_questions: List[SubQuestion]) -> float:
    # Heuristics: number of entities, temporal range, comparison words, ambiguity markers
    # Returns 0.0 (simple) → 1.0 (complex)
    # 0.0-0.3: 2 sub-questions, 0.3-0.7: 4-5, 0.7-1.0: up to 8
```

**Coding agent prompt:**
```
Create graph/nodes/planner.py.
- Async node function: async def planner_node(state: QueryMindState) -> dict
- Call planner_chain from Phase 1
- Implement compute_complexity() that scores query complexity based on:
  number of distinct entities, presence of comparison words, temporal markers, causal language
- Cap sub_questions list to complexity-based max (2/5/8)
- Return partial state update dict (only keys this node sets)
- Log an AgentTrace entry with start_time, end_time, tokens_used, node_name="planner"
```

---

### Step 2.3 — Parallel Retriever Nodes (The Core Challenge)

**File:** `graph/nodes/retriever.py`

**What to build:** A node factory that creates one retriever node per sub-question, all running in parallel using LangGraph's fan-out pattern.

**This is the hardest step in the project.** LangGraph fan-out requires:
1. A **routing function** that maps each sub-question to a named node
2. Multiple **retriever node instances** (one per question)
3. A **fan-in node** that waits for all retrievers to finish and merges results

```python
# The pattern you'll implement:
def route_to_retrievers(state: QueryMindState) -> List[Send]:
    return [
        Send("retriever_node", {"sub_question": sq, "session_id": state["session_id"]})
        for sq in state["sub_questions"]
    ]
```

**Key concept:** `Send` objects in LangGraph allow you to dynamically spawn parallel branches at runtime. This is how Google-style query fan-out works.

**Production challenge you'll hit:** What if one retriever fails? The graph hangs. Fix: wrap each retriever with a timeout + fallback that returns an empty result with an error flag rather than raising.

**Coding agent prompt:**
```
Create graph/nodes/retriever.py.
- Implement retriever_node(state) that runs web_search + semantic_search tools from Phase 1
- Implement route_to_retrievers(state) -> List[Send] for dynamic fan-out
- Each retriever runs independently with a 15-second asyncio timeout
- On timeout/error: return RetrievalResult with status="failed", error=str(e) — never raise
- Append AgentTrace for each retriever execution with sub_question_id field
```

---

### Step 2.4 — Conflict Detection & Arbitration Node

**File:** `graph/nodes/arbitrator.py`

**What to build:** After all retrievers finish, compare their results. If two retrievers return contradicting claims about the same entity/fact, spawn an arbitration sub-process.

**What counts as a conflict:**
- Different dates for the same event
- Different numeric values (prices, stats, counts)
- Contradictory boolean facts ("X is available" vs "X was discontinued")

**How arbitration works:**
1. `conflict_detector` scans all `RetrievalResult` objects with an LLM call
2. For each conflict, `arbitrator_chain` reasons about source credibility, recency, and corroboration
3. Returns `ArbitrationResult` with winning claim + reasoning

**This is what makes your project unique.** Almost no portfolio projects handle agent disagreement.

**Coding agent prompt:**
```
Create graph/nodes/arbitrator.py.
- conflict_detector(results: List[RetrievalResult]) -> List[Conflict]
  Use Gemini Flash to scan result pairs for factual contradictions
  Return structured Conflict(entity, claim_a, source_a, claim_b, source_b)
- arbitrate_conflict(conflict: Conflict) -> ArbitrationResult
  Use Gemini Pro + arbitrator_chain from Phase 1
  Reasoning must cite: source domain authority, publication recency, corroboration count
- conditional edge: if no conflicts → go to synthesizer; if conflicts → arbitrate first
```

---

### Step 2.5 — Streaming Synthesizer Node

**File:** `graph/nodes/synthesizer.py`

**What to build:** A node that takes all (arbitrated) retrieval results and streams a grounded, cited answer. The key production requirement: **the answer starts streaming before all agents finish**, using progressive confidence markers.

**The streaming pattern:**
```python
async def synthesizer_node(state: QueryMindState):
    # Stream tokens via .astream_events()
    async for event in synthesizer_chain.astream_events(input, version="v2"):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            # Push token to SSE queue
            await token_queue.put(token)
```

**Progressive confidence:** Early in the stream, inject `[LOW CONFIDENCE]` markers when only 2/5 retrievers have finished. Update to `[HIGH CONFIDENCE]` as more results come in. This mimics exactly what Google AI Mode does.

---

### Step 2.6 — Critic Node + Human-in-the-Loop

**File:** `graph/nodes/critic.py`

**What to build:** After synthesis, the critic scores the answer. If `confidence < 0.6` OR `flags` contains `"medical"` / `"legal"` / `"financial"`, the graph **pauses and waits for human approval** before returning.

**Key concept:** LangGraph `interrupt()` — this is how production systems implement human-in-the-loop. The graph checkpoints itself, pauses, and can be resumed later with human input.

```python
from langgraph.types import interrupt

async def critic_node(state: QueryMindState):
    score = await critic_chain.ainvoke(state["final_answer"])
    if score.confidence < 0.6 or any(f in score.flags for f in SENSITIVE_TOPICS):
        human_feedback = interrupt({
            "reason": "low_confidence",
            "answer": state["final_answer"],
            "score": score
        })
        # Graph pauses here. Resumes when .invoke() is called again with human_feedback
    return {"confidence_score": score, "requires_human_review": False}
```

---

### Step 2.7 — Memory & Cache Layer

**File:** `graph/checkpointer.py` and `core/memory/session_store.py`

**What to build:**
1. **LangGraph Checkpointer** — SQLite-backed (dev) / PostgreSQL-backed (prod) persistence so any graph run can be resumed after a crash or timeout
2. **Semantic Query Cache** — before the planner runs, check if a semantically similar query was answered in this session. If yes, reuse cached retrieval results and skip straight to synthesis.

**Key concepts:**
- `SqliteSaver` from `langgraph.checkpoint.sqlite`
- Graph `config={"configurable": {"thread_id": session_id}}`
- `graph.get_state(config)` — inspect any paused graph
- `graph.update_state(config, values)` — inject human feedback into a paused graph

**Coding agent prompt:**
```
Create core/memory/session_store.py.
- SessionStore class backed by SQLite
- store_result(session_id, query_hash, result) and get_cached(session_id, query)
- Semantic similarity check: embed the incoming query, compare cosine similarity to stored queries
- Cache hit threshold: 0.92 cosine similarity
- Cache TTL: 1 hour
Also create graph/checkpointer.py wrapping SqliteSaver with a factory function get_checkpointer(env).
```

---

### Step 2.8 — Assemble the Full Graph

**File:** `graph/query_mind_graph.py`

**What to build:** Wire all nodes into one `StateGraph` with correct edges and compile it.

```python
from langgraph.graph import StateGraph, START, END

def build_graph():
    g = StateGraph(QueryMindState)

    g.add_node("planner", planner_node)
    g.add_node("retriever_node", retriever_node)
    g.add_node("arbitrator", arbitrator_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("critic", critic_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_to_retrievers, ["retriever_node"])
    g.add_edge("retriever_node", "arbitrator")
    g.add_conditional_edges("arbitrator", route_to_synthesis, {
        "synthesize": "synthesizer",
        "re_retrieve": "planner"   # Loop back if arbitration fails
    })
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "approved": END,
        "human_review": END        # Graph paused via interrupt()
    })

    return g.compile(checkpointer=get_checkpointer("dev"))
```

**Production challenge you'll hit:** Cycle detection. The `arbitrator → planner` re-retrieval loop can run forever. Fix: add a `retry_count` field to state and a max-retries guard in the routing function.

---

### ✅ Phase 2 Checkpoint
- [ ] Full graph executes end-to-end on a test query
- [ ] Parallel retrievers confirmed running concurrently (check timestamps in AgentTrace)
- [ ] Conflict detection triggers correctly on a known contradictory topic (try: "Python 2 vs Python 3 release dates")
- [ ] Human-in-the-loop interrupt works: graph pauses, state is inspectable, resumes with `.update_state()`
- [ ] Cache hit confirmed: re-run same query, `cache_hits` is non-empty, latency drops by >50%
- [ ] Graph survives a simulated retriever timeout (kill Tavily call, confirm graceful error result)
- [ ] LangSmith trace shows full graph topology with per-node latency

---

## 🏗️ Phase 3 — Google ADK: Top-Level Orchestration (Days 17–21)

**Goal:** Wrap your LangGraph system inside a Google ADK agent. ADK becomes the entry point — it handles the conversation interface, Google Search grounding, and lifecycle management. Your LangGraph graph becomes one of ADK's tools.

**Core concepts you will learn:**
- ADK `Agent` class and `Runner`
- `FunctionTool` and `AgentTool`
- Google Search grounding (`google_search` built-in tool)
- ADK `Session` and `Memory` services
- Multi-turn conversation management
- ADK callbacks and lifecycle hooks
- Deploying to Vertex AI Agent Engine

---

### Step 3.1 — Wrap LangGraph Graph as ADK Tool

**File:** `adk/tools/querymind_tool.py`

**What to build:** Expose your entire LangGraph graph as a single `FunctionTool` that ADK's root agent can call.

```python
from google.adk.tools import FunctionTool

async def run_querymind(query: str, session_id: str) -> dict:
    """
    Runs the QueryMind multi-agent search pipeline.
    Use this for any research query requiring deep, multi-source analysis.
    Returns a cited answer with confidence score.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke({"original_query": query, "session_id": session_id}, config)
    return result["final_answer"].model_dump()

querymind_tool = FunctionTool(run_querymind)
```

**Key insight:** ADK's tool docstrings are used by the LLM to decide *when* to call the tool. Write them like you're writing prompts.

---

### Step 3.2 — Root Orchestrator Agent

**File:** `adk/orchestrator.py`

**What to build:** The ADK root agent that decides between calling QueryMind (deep research) vs Google Search grounding (quick factual lookup).

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

root_agent = LlmAgent(
    name="QueryMind Orchestrator",
    model="gemini-1.5-pro",
    description="Routes queries to appropriate search strategy",
    instruction="""
    You are an intelligent search orchestrator.
    For simple factual questions (dates, definitions, current values): use google_search directly.
    For complex research questions (analysis, comparisons, multi-step reasoning): use run_querymind.
    For follow-up questions in a conversation: check if context from prior turns already answers it.
    Always cite your sources in the final response.
    """,
    tools=[google_search, querymind_tool]
)
```

**Production challenge you'll hit:** The root agent sometimes calls QueryMind for simple questions (wasted cost). Fix: add a complexity pre-check with a Gemini Flash call that routes before the root agent sees the query.

---

### Step 3.3 — Google Search Grounding Integration

**File:** `adk/grounding.py`

**What to build:** Use ADK's native Google Search grounding to post-verify QueryMind's final answer. This adds a second layer of factual verification using Google's real-time index.

```python
from google.adk.tools import google_search
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

grounding_config = GenerateContentConfig(
    tools=[Tool(google_search=GoogleSearch())],
    temperature=0.0  # Deterministic for verification
)
```

**Key concept:** Google Search grounding in ADK works differently from Tavily — it doesn't return raw results. Instead, the model cites `[1]` style inline references and ADK returns the grounding metadata separately. Learn to parse `grounding_metadata.grounding_chunks`.

---

### Step 3.4 — ADK Session & Multi-turn Memory

**File:** `adk/session_manager.py`

**What to build:** Use ADK's `InMemorySessionService` (dev) and `VertexAiSessionService` (prod) to maintain conversation context across turns, so follow-up questions like "What about its performance?" work correctly.

```python
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

# Each conversation turn
session = await session_service.get_session(app_name="querymind", user_id=user_id, session_id=session_id)
# session.state contains the full conversation history + any state you set
```

**Integration point:** Connect ADK sessions to LangGraph checkpointer. When a follow-up arrives, load the prior LangGraph state and continue from there rather than re-running from scratch.

---

### Step 3.5 — ADK Callbacks & Lifecycle Hooks

**File:** `adk/callbacks.py`

**What to build:** Instrument every ADK agent call with custom callbacks for your observability dashboard.

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest):
    # Log: agent name, model, token estimate, timestamp
    pass

def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
    # Log: tokens used, latency, grounding metadata
    pass
```

---

### ✅ Phase 3 Checkpoint
- [ ] ADK root agent correctly routes simple vs complex queries (write 10 test cases)
- [ ] Google Search grounding adds citations to final answer
- [ ] Multi-turn conversation: "Tell me about LangGraph" → "Who built it?" resolves correctly
- [ ] ADK callbacks feed observability dashboard with real data
- [ ] End-to-end latency for a complex query: < 12 seconds (benchmark this)

---

## 📊 Phase 4 — Observability Dashboard (Days 22–24)

**Goal:** Build the live dashboard that shows every agent's execution state, latency, and cost. This is what separates a toy project from production engineering.

**File:** `observability/dashboard/app.py` (Streamlit)

### What to display

**Per-query view:**
- Timeline: each agent as a horizontal bar (start → end time), color-coded by status
- Token usage: bar chart per agent
- Cost estimate: `$0.000N per sub-question` breakdown
- Conflict log: which conflicts were detected + arbitration reasoning
- Cache hit indicator: which sub-questions were served from cache

**Session view:**
- Query history with latency trend
- Cache hit rate over time
- Average confidence score trend
- Human-in-the-loop intervention count

**Coding agent prompt:**
```
Create observability/dashboard/app.py using Streamlit.
- Connect to SQLite DB where AgentTrace records are stored
- Query view: Gantt-style chart using plotly for agent timeline
- Session view: line charts for latency + confidence trends
- Auto-refresh every 3 seconds using st.rerun() + time.sleep()
- Add a "replay query" button that re-runs a past query and compares results
```

---

## 🌐 Phase 5 — API & Streaming Endpoint (Days 25–26)

**Goal:** Expose QueryMind as a production-grade streaming API.

**File:** `api/server.py`

### Streaming endpoint (Server-Sent Events)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    async def event_generator():
        token_queue = asyncio.Queue()
        # Run graph in background, tokens pushed to queue
        asyncio.create_task(run_graph_with_streaming(request.query, token_queue))
        while True:
            token = await token_queue.get()
            if token is None:  # Sentinel value = done
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/query/{session_id}/state")
async def get_query_state(session_id: str):
    # Return current graph state for observability
    ...

@app.post("/query/{session_id}/resume")
async def resume_with_human_feedback(session_id: str, feedback: HumanFeedback):
    # Resume a paused (human-in-the-loop) graph
    ...
```

---

## 🧪 Phase 6 — Production Hardening (Days 27–30)

### What to implement

**Retry & Fallback Strategy**
- Every LLM call: `gemini-1.5-pro` with fallback to `gemini-1.5-flash` on timeout
- Every tool call: 3 retries with exponential backoff (tenacity)
- Every graph run: dead-letter queue for failed sessions

**Rate Limiting**
- Per-user query rate limit (token bucket algorithm)
- Global Gemini API quota tracking with circuit breaker

**Cost Controls**
- Pre-flight token estimation before running expensive Gemini Pro calls
- Automatic downgrade to Flash if estimated cost > threshold
- Daily spend cap with hard stop

**Integration Tests**
```
tests/integration/
├── test_full_pipeline.py         # End-to-end query test
├── test_parallel_retrievers.py   # Verify true concurrency
├── test_conflict_resolution.py   # Known contradictory test cases
├── test_cache_behavior.py        # Cache hit/miss scenarios
├── test_hitl_interrupt.py        # Human-in-the-loop flow
└── test_streaming_output.py      # Token stream correctness
```

---

## 🎤 Interview Talking Points

When you describe this project in interviews at Google, Anthropic, OpenAI, or any AI-first company, here are the 5 questions you'll be asked and what to say:

**Q: "How did you handle parallel agent execution?"**
> "I used LangGraph's `Send` API to implement dynamic fan-out — at runtime, the planner generates N `Send` objects based on query complexity, each spawning an independent retriever. State merging uses LangGraph's reducer pattern with `Annotated[List, operator.add]` to safely combine concurrent outputs without race conditions."

**Q: "What happens when two agents return contradicting information?"**
> "I built a dedicated Arbitration Agent that activates conditionally. It evaluates conflicts on three dimensions: source domain authority, publication recency, and corroboration count across other retrievers. The arbitration reasoning is surfaced in the observability dashboard."

**Q: "How do you handle failures in production?"**
> "Every retriever node has a 15-second timeout and returns a typed error result rather than raising. The synthesizer gracefully degrades — if only 3 of 5 retrievers succeeded, it synthesizes from those 3 and marks the answer with a lower confidence flag. The critic then routes low-confidence answers through human review via LangGraph's `interrupt()` mechanism."

**Q: "How does the memory/caching work?"**
> "Two layers: LangGraph's SQLite checkpointer persists every graph state for resumability, and a semantic query cache uses cosine similarity at 0.92 threshold to detect similar follow-up queries. Cache hits skip directly to synthesis and reduce latency by over 50%."

**Q: "Why Google ADK on top of LangGraph?"**
> "ADK handles the conversation layer and Google Search grounding natively, while LangGraph owns the complex stateful orchestration. The two are complementary — ADK routes simple queries to Google Search directly and complex ones to the LangGraph pipeline. This architecture mirrors how Google AI Mode likely operates: a fast routing layer in front of a heavier multi-agent backend."

---

## 📅 Full Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Phase 0: Setup | Day 1 | Working dev environment |
| Phase 1: LangChain | Days 2–6 | All tools, chains, parsers tested |
| Phase 2: LangGraph | Days 7–16 | Full working multi-agent graph |
| Phase 3: Google ADK | Days 17–21 | ADK orchestrator + grounding |
| Phase 4: Observability | Days 22–24 | Live Streamlit dashboard |
| Phase 5: API | Days 25–26 | Streaming FastAPI endpoint |
| Phase 6: Hardening | Days 27–30 | Tests, retries, cost controls |

**Total: 30 days of focused work (~3-4 hours/day)**

---

## 🚀 How to Use This File with a Coding Agent

Each step in this roadmap includes a **"Coding agent prompt"** block. The intended workflow is:

1. Open this file in VS Code
2. Navigate to the step you're working on
3. Copy the coding agent prompt
4. Paste it into Cursor / GitHub Copilot Chat / Cline / Claude
5. Review the generated code — **don't accept blindly**
6. Run the tests and hit the production challenge intentionally
7. Fix the challenge yourself before moving on

The production challenges are not optional. They are the actual learning. The goal is not to finish fast — it's to understand *why* production multi-agent systems fail and how to build them so they don't.

---

*QueryMind — Built to learn. Designed to impress.*