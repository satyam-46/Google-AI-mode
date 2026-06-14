"""Small ADK session boundary used by the orchestrator and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryMindSession:
    app_name: str
    user_id: str
    session_id: str
    state: dict[str, Any] = field(default_factory=dict)


class LocalSessionManager:
    """In-memory dev session manager mirroring the ADK session-service shape."""

    def __init__(self, app_name: str = "querymind") -> None:
        self.app_name = app_name
        self._sessions: dict[tuple[str, str], QueryMindSession] = {}

    async def get_session(self, user_id: str, session_id: str) -> QueryMindSession:
        key = (user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = QueryMindSession(self.app_name, user_id, session_id)
        return self._sessions[key]

    async def update_state(self, user_id: str, session_id: str, values: dict[str, Any]) -> QueryMindSession:
        session = await self.get_session(user_id, session_id)
        session.state.update(values)
        return session

    async def has_context(self, user_id: str, session_id: str) -> bool:
        session = await self.get_session(user_id, session_id)
        return bool(session.state)


__all__ = ["LocalSessionManager", "QueryMindSession"]

