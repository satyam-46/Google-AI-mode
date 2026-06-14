"""ADK lifecycle callbacks for observability."""

from __future__ import annotations

import time
from typing import Any


_EVENTS: list[dict[str, Any]] = []


def record_event(event: str, **details: Any) -> dict[str, Any]:
    payload = {"event": event, "timestamp_ms": int(time.time() * 1000), **details}
    _EVENTS.append(payload)
    return payload


def get_events() -> list[dict[str, Any]]:
    return list(_EVENTS)


def clear_events() -> None:
    _EVENTS.clear()


def before_model_callback(context: Any, request: Any) -> None:
    record_event(
        "before_model",
        agent_name=_context_attr(context, "agent_name"),
        model=_request_model(request),
    )
    return None


def after_model_callback(context: Any, response: Any) -> None:
    record_event(
        "after_model",
        agent_name=_context_attr(context, "agent_name"),
        grounding_chunks=_grounding_chunk_count(response),
    )
    return None


def on_model_error_callback(context: Any, request: Any, error: Exception) -> None:
    record_event(
        "model_error",
        agent_name=_context_attr(context, "agent_name"),
        model=_request_model(request),
        error_type=type(error).__name__,
        message=str(error),
    )
    return None


def _context_attr(context: Any, key: str) -> str:
    value = getattr(context, key, "")
    return str(value) if value is not None else ""


def _request_model(request: Any) -> str:
    config = getattr(request, "config", None)
    model = getattr(request, "model", None) or getattr(config, "model", None)
    return str(model) if model else ""


def _grounding_chunk_count(response: Any) -> int:
    metadata = getattr(response, "grounding_metadata", None)
    chunks = getattr(metadata, "grounding_chunks", None)
    return len(chunks) if chunks else 0


__all__ = [
    "after_model_callback",
    "before_model_callback",
    "clear_events",
    "get_events",
    "on_model_error_callback",
    "record_event",
]
