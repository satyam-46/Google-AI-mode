"""SQLite-backed observability trace records."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


class TraceStore:
    """Persist lightweight runtime trace records for dashboards and audits."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        default_path = Path(os.getenv("QUERYMIND_TRACE_DB", ".querymind_traces.sqlite3"))
        self.db_path = Path(db_path or default_path)
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    async def record(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload.setdefault("timestamp_ms", int(time.time() * 1000))
        async with self._lock:
            self._conn.execute(
                """
                insert into trace_records(
                    timestamp_ms, session_id, run_id, event, node, tokens_used, latency_ms, cost_usd, payload_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("timestamp_ms"),
                    payload.get("session_id", ""),
                    payload.get("run_id", ""),
                    payload.get("event", payload.get("name", "")),
                    payload.get("node", payload.get("name", "")),
                    int(payload.get("tokens_used") or 0),
                    int(payload.get("latency_ms") or payload.get("duration_ms") or 0),
                    float(payload.get("cost_usd") or 0.0),
                    json.dumps(payload, ensure_ascii=True),
                ),
            )
            self._conn.commit()
        return payload

    async def list_records(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._conn.execute(
                """
                select timestamp_ms, session_id, run_id, event, node, tokens_used, latency_ms, cost_usd, payload_json
                from trace_records
                order by timestamp_ms desc, id desc
                limit ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "timestamp_ms": row["timestamp_ms"],
                "session_id": row["session_id"],
                "run_id": row["run_id"],
                "event": row["event"],
                "node": row["node"],
                "tokens_used": row["tokens_used"],
                "latency_ms": row["latency_ms"],
                "cost_usd": row["cost_usd"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    async def record_graph_state(self, state: dict[str, Any]) -> None:
        session_id = str(state.get("session_id", ""))
        run_id = str(state.get("run_id", ""))
        for trace in state.get("agent_traces") or []:
            start_ms = int(trace.get("start_ms") or 0)
            end_ms = int(trace.get("end_ms") or start_ms)
            node = str(trace.get("name", ""))
            await self.record(
                {
                    "timestamp_ms": end_ms or int(time.time() * 1000),
                    "session_id": session_id,
                    "run_id": run_id,
                    "event": "agent_trace",
                    "node": node,
                    "name": node,
                    "tokens_used": int(trace.get("tokens_used") or 0),
                    "latency_ms": max(0, end_ms - start_ms),
                    "details": trace.get("details") or {},
                }
            )

    def _setup(self) -> None:
        self._conn.execute(
            """
            create table if not exists trace_records (
                id integer primary key autoincrement,
                timestamp_ms integer not null,
                session_id text not null default '',
                run_id text not null default '',
                event text not null default '',
                node text not null default '',
                tokens_used integer not null default 0,
                latency_ms integer not null default 0,
                cost_usd real not null default 0,
                payload_json text not null
            )
            """
        )
        self._conn.commit()


_DEFAULT_STORE: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = TraceStore()
    return _DEFAULT_STORE


def record_trace(record: dict[str, Any]) -> dict[str, Any]:
    """Synchronously persist a trace record when no event loop is active."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(get_trace_store().record(record))

    loop.create_task(get_trace_store().record(record))
    return record
