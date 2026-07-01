from __future__ import annotations

import json

from fastapi.testclient import TestClient

import api.server as api_server
from core.hardening import reset_hardening_singletons
from core.memory.session_store import SessionStore
from graph.nodes.cache import get_session_store


def _sse_data_frames(text: str) -> list[str]:
    return [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]


def test_stream_then_state_api_pipeline_is_inspectable(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYMIND_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.setenv("QUERYMIND_TRACE_DB", str(tmp_path / "traces.sqlite3"))
    monkeypatch.setenv("QUERYMIND_DEAD_LETTER_DB", str(tmp_path / "dead_letters.sqlite3"))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    reset_hardening_singletons()
    get_session_store.cache_clear()
    api_server._session_store = SessionStore(db_path=tmp_path / "api_sessions.sqlite3")

    client = TestClient(api_server.app)
    response = client.post(
        "/query/stream",
        json={"query": "What is an unknown local-only integration topic?", "session_id": "phase6-stream-state"},
    )

    assert response.status_code == 200
    frames = _sse_data_frames(response.text)
    payloads = [json.loads(frame) for frame in frames if frame != "[DONE]"]
    assert any(payload["event"] == "node_start" for payload in payloads)
    assert any(payload["event"] == "token" for payload in payloads)
    assert frames[-1] == "[DONE]"

    state_response = client.get("/query/phase6-stream-state/state")

    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["session_id"] == "phase6-stream-state"
    assert state_payload["state"]["session_id"] == "phase6-stream-state"
    assert "final_answer" in state_payload["state"]

    get_session_store.cache_clear()
