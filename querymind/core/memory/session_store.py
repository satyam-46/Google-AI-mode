from typing import Dict, Any
import asyncio


class SessionStore:
    """Simple in-memory session store for QueryMind state."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def save(self, session_id: str, state: Dict[str, Any]) -> None:
        async with self._lock:
            self._store[session_id] = state

    async def load(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            return self._store.get(session_id, {})
