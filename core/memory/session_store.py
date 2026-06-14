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

from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
        self._embeddings = self._create_embeddings()
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
        query_embedding = await self._embed_query(query)
        async with self._lock:
            now = int(time.time())
            self._conn.execute(
                """
                insert into query_cache(
                    session_id, query_hash, query_text, query_tokens_json, query_embedding_json, result_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(session_id, query_hash) do update set
                    query_text = excluded.query_text,
                    query_tokens_json = excluded.query_tokens_json,
                    query_embedding_json = excluded.query_embedding_json,
                    result_json = excluded.result_json,
                    created_at = excluded.created_at
                """,
                (
                    session_id,
                    self._query_hash(query),
                    query,
                    json.dumps(dict(_tokens(query)), ensure_ascii=True),
                    json.dumps(query_embedding, ensure_ascii=True) if query_embedding else None,
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
                select query_hash, query_text, query_tokens_json, query_embedding_json, result_json
                from query_cache
                where session_id = ? and created_at >= ?
                """,
                (session_id, cutoff),
            ).fetchall()

        query_tokens = _tokens(query)
        query_embedding = await self._embed_query(query)
        best: tuple[float, str, sqlite3.Row] | None = None
        for row in rows:
            cached_embedding = json.loads(row["query_embedding_json"]) if row["query_embedding_json"] else None
            if query_embedding and cached_embedding:
                similarity = _vector_cosine(query_embedding, cached_embedding)
                method = "embedding"
            else:
                cached_tokens = Counter(json.loads(row["query_tokens_json"]))
                similarity = _cosine(query_tokens, cached_tokens)
                method = "lexical"
            if best is None or similarity > best[0]:
                best = (similarity, method, row)

        if best is None or best[0] < threshold:
            return None

        result = json.loads(best[2]["result_json"])
        result["cache_hit"] = {
            "query_hash": best[2]["query_hash"],
            "query_text": best[2]["query_text"],
            "similarity": best[0],
            "method": best[1],
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
                query_embedding_json text,
                result_json text not null,
                created_at integer not null,
                primary key (session_id, query_hash)
            )
            """
        )
        columns = {
            row["name"]
            for row in self._conn.execute("pragma table_info(query_cache)").fetchall()
        }
        if "query_embedding_json" not in columns:
            self._conn.execute("alter table query_cache add column query_embedding_json text")
        self._conn.commit()

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    @staticmethod
    def _create_embeddings() -> GoogleGenerativeAIEmbeddings | None:
        if not os.getenv("GOOGLE_API_KEY"):
            return None
        try:
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception:
            return None

    async def _embed_query(self, query: str) -> list[float] | None:
        if self._embeddings is None:
            return None
        try:
            return await self._embeddings.aembed_query(query)
        except Exception:
            return None


def _tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text.lower()))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _vector_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
