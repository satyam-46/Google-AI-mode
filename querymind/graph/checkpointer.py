"""Checkpointer factory for LangGraph runs."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


def get_checkpointer(env: str = "dev", db_path: str | Path | None = None) -> Any:
    """Return a LangGraph checkpointer.

    Dev defaults to SQLite when `langgraph-checkpoint-sqlite` is installed.
    Tests can pass `env="test"` for an isolated in-memory saver.
    """
    if env == "test":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    path = Path(db_path or os.getenv("QUERYMIND_CHECKPOINT_DB", ".querymind_checkpoints.sqlite3"))
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    except Exception:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
