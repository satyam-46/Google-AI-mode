from __future__ import annotations

import json

from fastapi.testclient import TestClient

import api.server as api_server


class FakeSessionStore:
    def __init__(self, state: dict | None = None) -> None:
        self.state = state or {}
        self.saved: list[tuple[str, dict]] = []

    async def load(self, session_id: str) -> dict:
        return self.state

    async def save(self, session_id: str, state: dict) -> None:
        self.saved.append((session_id, state))
        self.state = state


def _sse_data_frames(text: str) -> list[str]:
    return [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]


def test_stream_event_normalization_uses_public_schema():
    start = api_server.normalize_stream_event(
        {"event": "on_chain_start", "node": "planner", "session_id": "internal"},
        session_id="public-session",
        sequence=1,
    )
    token = api_server.normalize_stream_event(
        {"event": "token", "token": "hello", "session_id": "internal"},
        session_id="public-session",
        sequence=2,
    )

    assert start == {
        "event": "node_start",
        "session_id": "public-session",
        "sequence": 1,
        "node": "planner",
        "data": {"raw_event": "on_chain_start"},
    }
    assert token["event"] == "token"
    assert token["token"] == "hello"
    assert token["sequence"] == 2


def test_stream_endpoint_emits_ordered_sse_events_and_done(monkeypatch):
    store = FakeSessionStore()

    async def fake_stream_query_graph_events(query: str, session_id: str):
        yield {"event": "on_chain_start", "node": "planner", "session_id": session_id}
        yield {"event": "token", "token": "answer", "session_id": session_id}
        yield {"event": "on_chain_end", "node": "planner", "session_id": session_id}

    monkeypatch.setattr(api_server, "_session_store", store)
    monkeypatch.setattr(api_server, "stream_query_graph_events", fake_stream_query_graph_events)
    monkeypatch.setattr(
        api_server,
        "get_projected_query_graph_state",
        lambda session_id: {"session_id": session_id, "final_answer": {"answer_text": "answer"}},
    )

    client = TestClient(api_server.app)
    response = client.post("/query/stream", json={"query": "stream this", "session_id": "stream-session"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _sse_data_frames(response.text)
    payloads = [json.loads(frame) for frame in frames[:-1]]

    assert [payload["event"] for payload in payloads] == ["node_start", "token", "node_end"]
    assert [payload["sequence"] for payload in payloads] == [1, 2, 3]
    assert payloads[1]["token"] == "answer"
    assert frames[-1] == "[DONE]"
    assert store.saved[-1][0] == "stream-session"


def test_stream_endpoint_emits_structured_error_event(monkeypatch):
    store = FakeSessionStore()

    async def failing_stream_query_graph_events(query: str, session_id: str):
        raise RuntimeError("stream boom")
        yield  # pragma: no cover

    monkeypatch.setattr(api_server, "_session_store", store)
    monkeypatch.setattr(api_server, "stream_query_graph_events", failing_stream_query_graph_events)

    client = TestClient(api_server.app)
    response = client.post("/query/stream", json={"query": "fail stream", "session_id": "failed-stream"})

    frames = _sse_data_frames(response.text)
    error_payload = json.loads(frames[0])

    assert error_payload["event"] == "error"
    assert error_payload["error"] == {
        "stage": "stream",
        "type": "RuntimeError",
        "message": "stream boom",
    }
    assert frames[-1] == "[DONE]"
    assert store.saved[-1][1]["error"] == "stream boom"


def test_state_endpoint_returns_session_store_fallback(monkeypatch):
    store = FakeSessionStore({"session_id": "stored-session", "original_query": "saved"})

    monkeypatch.setattr(api_server, "_session_store", store)
    monkeypatch.setattr(api_server, "get_query_graph_state", lambda session_id: None)

    client = TestClient(api_server.app)
    response = client.get("/query/stored-session/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "stored-session"
    assert payload["state"]["original_query"] == "saved"
    assert payload["next"] == []


def test_state_endpoint_returns_structured_not_found(monkeypatch):
    monkeypatch.setattr(api_server, "_session_store", FakeSessionStore({}))
    monkeypatch.setattr(api_server, "get_query_graph_state", lambda session_id: None)

    client = TestClient(api_server.app)
    response = client.get("/query/missing/state")

    assert response.status_code == 404
    assert response.json()["error"]["stage"] == "state_lookup"
    assert response.json()["error"]["type"] == "NotFound"


def test_resume_endpoint_wraps_state_in_response_model(monkeypatch):
    store = FakeSessionStore()

    async def fake_resume_query_graph(session_id: str, feedback: dict):
        return {
            "session_id": session_id,
            "requires_human_review": False,
            "final_answer": {
                "answer_text": "Approved answer.",
                "citations": [{"url": "https://example.com"}],
                "confidence": 0.88,
            },
            "human_feedback": feedback,
        }

    monkeypatch.setattr(api_server, "_session_store", store)
    monkeypatch.setattr(api_server, "resume_query_graph", fake_resume_query_graph)

    client = TestClient(api_server.app)
    response = client.post("/query/resume-session/resume", json={"feedback": {"approved": True}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "resume-session"
    assert payload["answer"] == "Approved answer."
    assert payload["confidence"] == 0.88
    assert payload["requires_human_review"] is False
    assert store.saved[-1][0] == "resume-session"


def test_resume_endpoint_returns_structured_conflict(monkeypatch):
    async def failing_resume_query_graph(session_id: str, feedback: dict):
        raise RuntimeError("not paused")

    monkeypatch.setattr(api_server, "resume_query_graph", failing_resume_query_graph)

    client = TestClient(api_server.app)
    response = client.post("/query/not-paused/resume", json={"feedback": {"approved": True}})

    assert response.status_code == 409
    assert response.json()["error"] == {
        "stage": "resume",
        "type": "RuntimeError",
        "message": "not paused",
    }
