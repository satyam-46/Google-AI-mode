from typing import Any


def get_checkpointer(env: str = "dev") -> Any:
    """Return a checkpointer object for the given environment.

    This is a stub. Replace with LangGraph SqliteSaver/Sqlalchemy-backed implementation.
    """
        # For dev we'll return a lightweight in-memory placeholder
        class _InMemorySaver:
            def __init__(self):
                self._data = {}

            def save(self, key, value):
                self._data[key] = value

            def load(self, key, default=None):
                return self._data.get(key, default)

        return _InMemorySaver()
