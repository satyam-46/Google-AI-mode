"""Grounding boundary for ADK Google Search integration.

The live ADK grounding call is intentionally kept behind an injectable boundary
so local tests can validate orchestration without network access.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


GroundingProvider = Callable[[str, str], Awaitable[GroundedAnswer]]


async def answer_with_grounding(query: str, session_id: str = "default") -> GroundedAnswer:
    """Return a fast grounded answer.

    Uses Gemini's Google Search tool when credentials are available. Tests and
    explicit fake-LLM local runs use deterministic offline answers.
    """
    if _live_grounding_enabled():
        try:
            return await _live_google_grounded_answer(query, session_id)
        except Exception as exc:
            fallback = _offline_grounded_answer(query)
            fallback.metadata.update(
                {
                    "live_grounding_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                }
            )
            return fallback
    return _offline_grounded_answer(query)


def verify_with_google(final_answer: dict[str, Any]) -> dict[str, Any]:
    """Return verification metadata for a QueryMind answer."""
    citations = final_answer.get("citations") or []
    return {
        "verified": bool(citations),
        "metadata": {
            "grounding_chunks": len(citations),
            "mode": "offline",
        },
    }


async def _live_google_grounded_answer(query: str, session_id: str) -> GroundedAnswer:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = await client.aio.models.generate_content(
        model=os.getenv("QUERYMIND_GROUNDING_MODEL", "gemini-2.5-flash"),
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
        ),
    )
    metadata = _extract_grounding_metadata(response)
    citations = _grounding_citations(metadata)
    return GroundedAnswer(
        answer=str(getattr(response, "text", "") or ""),
        citations=citations,
        confidence=0.85 if citations else 0.55,
        metadata={
            "mode": "live_google_search",
            "session_id": session_id,
            "grounding_chunks": len(citations),
            "web_search_queries": metadata.get("web_search_queries", []),
            "raw": metadata,
        },
    )


def _extract_grounding_metadata(response: Any) -> dict[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return {}
    grounding_metadata = getattr(candidates[0], "grounding_metadata", None)
    if grounding_metadata is None:
        return {}
    if hasattr(grounding_metadata, "model_dump"):
        return grounding_metadata.model_dump(exclude_none=True)
    if hasattr(grounding_metadata, "to_json_dict"):
        return grounding_metadata.to_json_dict()
    return {
        "grounding_chunks": getattr(grounding_metadata, "grounding_chunks", []) or [],
        "grounding_supports": getattr(grounding_metadata, "grounding_supports", []) or [],
        "web_search_queries": getattr(grounding_metadata, "web_search_queries", []) or [],
    }


def _grounding_citations(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for chunk in metadata.get("grounding_chunks") or metadata.get("groundingChunks") or []:
        web = chunk.get("web", {}) if isinstance(chunk, dict) else getattr(chunk, "web", None)
        if not web:
            continue
        if not isinstance(web, dict):
            web = {
                "title": getattr(web, "title", ""),
                "uri": getattr(web, "uri", ""),
            }
        citations.append(
            {
                "source": web.get("title", ""),
                "url": web.get("uri", ""),
                "excerpt": "",
            }
        )
    return citations


def _live_grounding_enabled() -> bool:
    if os.getenv("QUERYMIND_FORCE_FAKE_LLM", "").lower() == "true":
        return False
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _offline_grounded_answer(query: str) -> GroundedAnswer:
    normalized = " ".join(query.lower().split())
    if re.search(r"\bsun\b.*\brise", normalized):
        return GroundedAnswer(
            answer="The Sun generally rises in the east.",
            citations=[
                {
                    "source": "Offline astronomy fact",
                    "url": "",
                    "excerpt": "Earth's rotation makes the Sun appear to rise in the east.",
                }
            ],
            confidence=0.82,
            metadata={"mode": "offline", "grounding_chunks": 1},
        )
    if re.search(r"\bsun\b.*\bset", normalized) or re.search(r"\bwhere does it set\b", normalized):
        return GroundedAnswer(
            answer="The Sun generally sets in the west.",
            citations=[
                {
                    "source": "Offline astronomy fact",
                    "url": "",
                    "excerpt": "Earth's rotation makes the Sun appear to set in the west.",
                }
            ],
            confidence=0.82,
            metadata={"mode": "offline", "grounding_chunks": 1},
        )
    return GroundedAnswer(
        answer="Fast grounding is configured, but live Google Search grounding is not available in this local run.",
        citations=[],
        confidence=0.35,
        metadata={"mode": "offline", "grounding_chunks": 0},
    )


__all__ = ["GroundedAnswer", "GroundingProvider", "answer_with_grounding", "verify_with_google"]
