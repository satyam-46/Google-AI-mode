"""Run initialization node."""

from __future__ import annotations

import time
from typing import Any

from graph.state import QueryMindState


async def initialize_node(state: QueryMindState) -> dict[str, Any]:
    return {
        "started_at_ms": int(time.time() * 1000),
        "retrieval_results": [],
        "retriever_errors": [],
        "agent_traces": [],
        "cache_hits": [],
        "retry_count": state.get("retry_count", 0),
    }
