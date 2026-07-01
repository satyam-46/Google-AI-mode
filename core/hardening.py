"""Production hardening primitives for QueryMind."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RateLimitExceededError(RuntimeError):
    pass


class CircuitOpenError(RuntimeError):
    pass


class BudgetExceededError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """Cheap pre-flight token estimate for routing and spend controls."""
    stripped = " ".join(text.split())
    if not stripped:
        return 0
    word_estimate = math.ceil(len(stripped.split()) * 1.35)
    char_estimate = math.ceil(len(stripped) / 4)
    return max(1, max(word_estimate, char_estimate))


@dataclass
class TokenBucket:
    capacity: float
    refill_rate_per_second: float
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Per-key token bucket limiter."""

    def __init__(self, capacity: int = 30, refill_rate_per_minute: float = 30.0) -> None:
        self.capacity = float(capacity)
        self.refill_rate_per_second = float(refill_rate_per_minute) / 60.0
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, key: str, cost: float = 1.0, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(
                capacity=self.capacity,
                refill_rate_per_second=self.refill_rate_per_second,
                tokens=self.capacity,
                updated_at=now,
            )
            self._buckets[key] = bucket

        elapsed = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_rate_per_second)
        bucket.updated_at = now
        if bucket.tokens < cost:
            return False
        bucket.tokens -= cost
        return True


class CircuitBreaker:
    """Small failure-count circuit breaker for upstream model/search protection."""

    def __init__(self, failure_threshold: int = 5, reset_seconds: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.failure_count = 0
        self.opened_at: float | None = None

    def allow_request(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if self.opened_at is None:
            return True
        if now - self.opened_at >= self.reset_seconds:
            self.failure_count = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = now


class CostController:
    """Pre-flight cost estimation, model downgrade, and daily spend cap."""

    def __init__(
        self,
        daily_spend_cap_usd: float = 5.0,
        pro_call_threshold_usd: float = 0.02,
        flash_cost_per_1k_tokens: float = 0.00035,
        pro_cost_per_1k_tokens: float = 0.0035,
    ) -> None:
        self.daily_spend_cap_usd = daily_spend_cap_usd
        self.pro_call_threshold_usd = pro_call_threshold_usd
        self.flash_cost_per_1k_tokens = flash_cost_per_1k_tokens
        self.pro_cost_per_1k_tokens = pro_cost_per_1k_tokens
        self._spent_by_day: dict[str, float] = {}

    def estimate_cost(self, tokens: int, model: str = "gemini-2.5-flash") -> float:
        rate = self.pro_cost_per_1k_tokens if _is_pro_model(model) else self.flash_cost_per_1k_tokens
        return tokens / 1000 * rate

    def choose_model(self, requested_model: str, tokens: int) -> str:
        if _is_pro_model(requested_model) and self.estimate_cost(tokens, requested_model) > self.pro_call_threshold_usd:
            return "gemini-2.5-flash"
        return requested_model

    def ensure_budget(self, estimated_cost_usd: float, now: float | None = None) -> None:
        day = _day_key(now)
        spent = self._spent_by_day.get(day, 0.0)
        if spent + estimated_cost_usd > self.daily_spend_cap_usd:
            raise BudgetExceededError(
                f"Estimated cost ${estimated_cost_usd:.6f} would exceed daily cap ${self.daily_spend_cap_usd:.2f}"
            )

    def record_spend(self, cost_usd: float, now: float | None = None) -> None:
        day = _day_key(now)
        self._spent_by_day[day] = self._spent_by_day.get(day, 0.0) + max(0.0, cost_usd)

    def preflight(self, query: str, requested_model: str = "gemini-2.5-pro") -> dict[str, Any]:
        tokens = estimate_tokens(query)
        selected_model = self.choose_model(requested_model, tokens)
        estimated_cost = self.estimate_cost(tokens, selected_model)
        self.ensure_budget(estimated_cost)
        return {
            "estimated_tokens": tokens,
            "requested_model": requested_model,
            "selected_model": selected_model,
            "estimated_cost_usd": round(estimated_cost, 8),
            "downgraded": selected_model != requested_model,
        }


class DeadLetterStore:
    """SQLite-backed failed-session store."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        default_path = Path(os.getenv("QUERYMIND_DEAD_LETTER_DB", ".querymind_dead_letters.sqlite3"))
        self.db_path = Path(db_path or default_path)
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    async def record_failure(
        self,
        *,
        session_id: str,
        query: str,
        stage: str,
        error: Exception | dict[str, Any] | str,
        state: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(error, Exception):
            error_payload = {"type": type(error).__name__, "message": str(error)}
        elif isinstance(error, dict):
            error_payload = dict(error)
        else:
            error_payload = {"type": "RuntimeError", "message": str(error)}

        async with self._lock:
            self._conn.execute(
                """
                insert into dead_letters(session_id, query_text, stage, error_json, state_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    query,
                    stage,
                    json.dumps(error_payload, ensure_ascii=True),
                    json.dumps(state or {}, ensure_ascii=True),
                    int(time.time()),
                ),
            )
            self._conn.commit()

    async def list_failures(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._conn.execute(
                """
                select session_id, query_text, stage, error_json, state_json, created_at
                from dead_letters
                order by created_at desc, id desc
                limit ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "query": row["query_text"],
                "stage": row["stage"],
                "error": json.loads(row["error_json"]),
                "state": json.loads(row["state_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _setup(self) -> None:
        self._conn.execute(
            """
            create table if not exists dead_letters (
                id integer primary key autoincrement,
                session_id text not null,
                query_text text not null,
                stage text not null,
                error_json text not null,
                state_json text not null,
                created_at integer not null
            )
            """
        )
        self._conn.commit()


_RATE_LIMITER: TokenBucketRateLimiter | None = None
_CIRCUIT_BREAKER: CircuitBreaker | None = None
_COST_CONTROLLER: CostController | None = None
_DEAD_LETTER_STORE: DeadLetterStore | None = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        _RATE_LIMITER = TokenBucketRateLimiter(
            capacity=int(os.getenv("QUERYMIND_RATE_LIMIT_BURST", "30")),
            refill_rate_per_minute=float(os.getenv("QUERYMIND_RATE_LIMIT_PER_MINUTE", "30")),
        )
    return _RATE_LIMITER


def get_circuit_breaker() -> CircuitBreaker:
    global _CIRCUIT_BREAKER
    if _CIRCUIT_BREAKER is None:
        _CIRCUIT_BREAKER = CircuitBreaker(
            failure_threshold=int(os.getenv("QUERYMIND_CIRCUIT_FAILURES", "5")),
            reset_seconds=float(os.getenv("QUERYMIND_CIRCUIT_RESET_SECONDS", "60")),
        )
    return _CIRCUIT_BREAKER


def get_cost_controller() -> CostController:
    global _COST_CONTROLLER
    if _COST_CONTROLLER is None:
        _COST_CONTROLLER = CostController(
            daily_spend_cap_usd=float(os.getenv("QUERYMIND_DAILY_SPEND_CAP_USD", "5.0")),
            pro_call_threshold_usd=float(os.getenv("QUERYMIND_PRO_CALL_THRESHOLD_USD", "0.02")),
        )
    return _COST_CONTROLLER


def get_dead_letter_store() -> DeadLetterStore:
    global _DEAD_LETTER_STORE
    if _DEAD_LETTER_STORE is None:
        _DEAD_LETTER_STORE = DeadLetterStore()
    return _DEAD_LETTER_STORE


def reset_hardening_singletons() -> None:
    global _RATE_LIMITER, _CIRCUIT_BREAKER, _COST_CONTROLLER, _DEAD_LETTER_STORE
    _RATE_LIMITER = None
    _CIRCUIT_BREAKER = None
    _COST_CONTROLLER = None
    _DEAD_LETTER_STORE = None


def _is_pro_model(model: str) -> bool:
    return "pro" in model.lower()


def _day_key(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(time.time() if now is None else now))

