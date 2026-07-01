from __future__ import annotations

import pytest

import api.server as api_server
from core.hardening import (
    BudgetExceededError,
    CircuitBreaker,
    CostController,
    DeadLetterStore,
    TokenBucketRateLimiter,
    estimate_tokens,
)


class FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def test_token_estimator_and_model_downgrade():
    controller = CostController(daily_spend_cap_usd=10.0, pro_call_threshold_usd=0.00001)
    tokens = estimate_tokens("Compare LangGraph and LangChain for production research workflows.")

    assert tokens > 0
    assert controller.choose_model("gemini-2.5-pro", tokens) == "gemini-2.5-flash"
    assert controller.choose_model("gemini-2.5-flash", tokens) == "gemini-2.5-flash"


def test_cost_controller_enforces_daily_spend_cap():
    controller = CostController(daily_spend_cap_usd=0.001)
    controller.record_spend(0.0009, now=1_700_000_000)

    with pytest.raises(BudgetExceededError):
        controller.ensure_budget(0.0002, now=1_700_000_000)


def test_token_bucket_rate_limiter_refills_over_time():
    limiter = TokenBucketRateLimiter(capacity=2, refill_rate_per_minute=60)

    assert limiter.allow("user", now=100.0)
    assert limiter.allow("user", now=100.0)
    assert not limiter.allow("user", now=100.0)
    assert limiter.allow("user", now=101.0)


def test_circuit_breaker_opens_and_half_resets():
    breaker = CircuitBreaker(failure_threshold=2, reset_seconds=10)

    assert breaker.allow_request(now=100)
    breaker.record_failure(now=100)
    assert breaker.allow_request(now=101)
    breaker.record_failure(now=101)
    assert not breaker.allow_request(now=102)
    assert breaker.allow_request(now=112)


@pytest.mark.asyncio
async def test_dead_letter_store_records_failed_sessions(tmp_path):
    store = DeadLetterStore(db_path=tmp_path / "dead_letters.sqlite3")

    await store.record_failure(
        session_id="failed-session",
        query="bad query",
        stage="graph_run",
        error=RuntimeError("boom"),
        state={"partial": True},
    )
    failures = await store.list_failures()

    assert failures[0]["session_id"] == "failed-session"
    assert failures[0]["stage"] == "graph_run"
    assert failures[0]["error"] == {"type": "RuntimeError", "message": "boom"}
    assert failures[0]["state"] == {"partial": True}


def test_api_request_controls_return_rate_limit(monkeypatch):
    limiter = TokenBucketRateLimiter(capacity=0, refill_rate_per_minute=0)

    monkeypatch.setattr(api_server, "get_rate_limiter", lambda: limiter)

    response = api_server.enforce_request_controls("hello", "session", FakeRequest({"x-user-id": "u1"}))

    assert response is not None
    assert response.status_code == 429


def test_api_request_controls_return_circuit_open(monkeypatch):
    class OpenBreaker:
        def allow_request(self):
            return False

    monkeypatch.setattr(api_server, "get_rate_limiter", lambda: TokenBucketRateLimiter())
    monkeypatch.setattr(api_server, "get_circuit_breaker", lambda: OpenBreaker())

    response = api_server.enforce_request_controls("hello", "session", FakeRequest())

    assert response is not None
    assert response.status_code == 503


def test_api_request_controls_return_budget_exceeded(monkeypatch):
    controller = CostController(daily_spend_cap_usd=0.0)

    monkeypatch.setattr(api_server, "get_rate_limiter", lambda: TokenBucketRateLimiter())
    monkeypatch.setattr(api_server, "get_circuit_breaker", lambda: CircuitBreaker())
    monkeypatch.setattr(api_server, "get_cost_controller", lambda: controller)

    response = api_server.enforce_request_controls("expensive query", "session", FakeRequest())

    assert response is not None
    assert response.status_code == 402
