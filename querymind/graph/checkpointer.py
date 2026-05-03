"""Checkpointer factory placeholder for the LangGraph phase."""

from __future__ import annotations

from typing import Any


def get_checkpointer(env: str = "dev") -> Any:
    """Return an in-memory checkpointer until the LangGraph phase starts."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except Exception:
        return _InMemorySaver()


class _InMemorySaver:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def save(self, key: str, value: Any) -> None:
        self._data[key] = value

    def load(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
