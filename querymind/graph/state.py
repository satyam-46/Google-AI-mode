from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel


class QueryMindState(TypedDict, total=False):
    original_query: str
    session_id: str

    # Planner
    sub_questions: List[dict]
    complexity_score: float

    # Retriever outputs
    retrieval_results: List[dict]

    # Arbitration
    conflicts_detected: List[dict]
    arbitration_results: List[dict]

    # Synthesis
    streaming_answer: Optional[str]
    final_answer: Optional[dict]

    # Critic
    confidence_score: Optional[dict]
    requires_human_review: bool

    # Observability / memory
    agent_traces: List[dict]
    total_tokens_used: int
    total_latency_ms: int


def example_state() -> QueryMindState:
    return {"original_query": "", "session_id": "", "agent_traces": [], "total_tokens_used": 0, "total_latency_ms": 0}


class AgentTrace(BaseModel):
    name: str
    start_ms: int
    end_ms: int
    details: Dict[str, Any] = {}
