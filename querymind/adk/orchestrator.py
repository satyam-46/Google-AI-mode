"""ADK orchestrator stub."""

from typing import Any


async def route_query(query: str, session_id: str) -> Any:
    """Decide whether to use quick grounding or full QueryMind pipeline.

    This is a stub for the ADK orchestrator.
    """
    return {"routed_to": "querymind", "session_id": session_id}
