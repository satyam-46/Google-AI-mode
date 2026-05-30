"""SQLite-backed session state and query cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class SessionStore:
    """Persist graph state and cache recent query results per session."""

    def __init__(self, db_path: str | Path | None = None, cache_ttl_seconds: int = 3600) -> None:
        default_path = Path(os.getenv("QUERYMIND_SESSION_DB", ".querymind_sessions.sqlite3"))
        self.db_path = Path(db_path or default_path)
        self.cache_ttl_seconds = cache_ttl_seconds
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    async def save(self, session_id: str, state: dict[str, Any]) -> None:
        async with self._lock:
            payload = json.dumps(state, ensure_ascii=True)
            now = int(time.time())
            self._conn.execute(
                """
                insert into sessions(session_id, state_json, updated_at)
                values (?, ?, ?)
                on conflict(session_id) do update set
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, payload, now),
            )
            self._conn.commit()

    async def load(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            row = self._conn.execute(
                "select state_json from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
            return json.loads(row["state_json"]) if row else {}

    async def store_result(self, session_id: str, query: str, result: dict[str, Any]) -> None:
        async with self._lock:
            now = int(time.time())
            self._conn.execute(
                """
                insert into query_cache(session_id, query_hash, query_text, query_tokens_json, result_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(session_id, query_hash) do update set
                    query_text = excluded.query_text,
                    query_tokens_json = excluded.query_tokens_json,
                    result_json = excluded.result_json,
                    created_at = excluded.created_at
                """,
                (
                    session_id,
                    self._query_hash(query),
                    query,
                    json.dumps(dict(_tokens(query)), ensure_ascii=True),
                    json.dumps(result, ensure_ascii=True),
                    now,
                ),
            )
            self._conn.commit()

    async def get_cached(self, session_id: str, query: str, threshold: float = 0.92) -> dict[str, Any] | None:
        async with self._lock:
            cutoff = int(time.time()) - self.cache_ttl_seconds
            rows = self._conn.execute(
                """
                select query_hash, query_text, query_tokens_json, result_json
                from query_cache
                where session_id = ? and created_at >= ?
                """,
                (session_id, cutoff),
            ).fetchall()

        query_tokens = _tokens(query)
        best: tuple[float, sqlite3.Row] | None = None
        for row in rows:
            cached_tokens = Counter(json.loads(row["query_tokens_json"]))
            similarity = _cosine(query_tokens, cached_tokens)
            if best is None or similarity > best[0]:
                best = (similarity, row)

        if best is None or best[0] < threshold:
            return None

        result = json.loads(best[1]["result_json"])
        result["cache_hit"] = {
            "query_hash": best[1]["query_hash"],
            "query_text": best[1]["query_text"],
            "similarity": best[0],
        }
        return result

    def _setup(self) -> None:
        self._conn.execute(
            """
            create table if not exists sessions (
                session_id text primary key,
                state_json text not null,
                updated_at integer not null
            )
            """
        )
        self._conn.execute(
            """
            create table if not exists query_cache (
                session_id text not null,
                query_hash text not null,
                query_text text not null,
                query_tokens_json text not null,
                result_json text not null,
                created_at integer not null,
                primary key (session_id, query_hash)
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def _tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text.lower()))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
