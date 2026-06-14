"""Deterministic route selection for the ADK orchestration layer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class QueryRoute(StrEnum):
    FAST = "fast"
    DEEP = "deep"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class RouteDecision:
    route: QueryRoute
    reason: str
    complexity_score: float
    matched_rules: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, object]:
        return {
            "route": self.route.value,
            "reason": self.reason,
            "complexity_score": self.complexity_score,
            "matched_rules": list(self.matched_rules),
        }


_FAST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("direct_where", re.compile(r"^\s*where\s+(is|are|does|do|did|was|were)\b", re.I)),
    ("direct_when", re.compile(r"^\s*when\s+(is|are|does|do|did|was|were)\b", re.I)),
    ("direct_who", re.compile(r"^\s*who\s+(is|are|was|were|built|created|founded)\b", re.I)),
    ("direct_what", re.compile(r"^\s*what\s+(is|are|was|were|does|do)\b", re.I)),
    ("definition", re.compile(r"^\s*(define|meaning of|what does .+ mean\b)", re.I)),
    ("capital", re.compile(r"\bcapital of\b", re.I)),
    ("simple_direction", re.compile(r"\b(rise|rises|set|sets)\b.*\b(direction|east|west|north|south)\b", re.I)),
)

_DEEP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("comparison", re.compile(r"\b(compare|versus|vs\.?|difference between|better than)\b", re.I)),
    ("analysis", re.compile(r"\b(analy[sz]e|why|how should|trade[- ]?offs?|pros and cons|evaluate)\b", re.I)),
    ("research", re.compile(r"\b(research|deep dive|multi[- ]?source|cite|citations|sources|evidence)\b", re.I)),
    ("current", re.compile(r"\b(latest|current|today|yesterday|this week|202[5-9]|price|stock|weather)\b", re.I)),
    ("recommendation", re.compile(r"\b(recommend|best|top|should I|which .* should)\b", re.I)),
    ("conflict", re.compile(r"\b(conflict|contradict|disagree|arbitrat|fact check|verify)\b", re.I)),
    ("sensitive", re.compile(r"\b(medical|legal|law|tax|financial|investment|invest|diagnosis|treatment)\b", re.I)),
)

_FOLLOW_UP_RE = re.compile(r"^\s*(what about|and|also|it|that|those|they|he|she|where does it|who built it)\b", re.I)


def route_query_text(query: str, has_session_context: bool = False) -> RouteDecision:
    """Choose fast grounding or the full QueryMind graph without another LLM call."""
    normalized = " ".join(query.strip().split())
    if not normalized:
        return RouteDecision(QueryRoute.CLARIFY, "Empty query needs clarification.", 0.0, ["empty"])

    tokens = re.findall(r"[a-z0-9]+", normalized.lower())
    matched: list[str] = []
    score = min(len(tokens) / 40.0, 0.35)

    for name, pattern in _DEEP_PATTERNS:
        if pattern.search(normalized):
            matched.append(name)
            score += 0.18

    fast_matches = [name for name, pattern in _FAST_PATTERNS if pattern.search(normalized)]
    if fast_matches:
        matched.extend(fast_matches)
        score -= 0.12

    if _FOLLOW_UP_RE.search(normalized):
        matched.append("follow_up")
        if not has_session_context:
            return RouteDecision(
                QueryRoute.CLARIFY,
                "Follow-up query has no usable session context.",
                max(0.2, min(score + 0.15, 1.0)),
                matched,
            )
        score += 0.1

    if normalized.count("?") > 1 or re.search(r"\b(and|or)\b", normalized, re.I) and len(tokens) > 10:
        matched.append("compound")
        score += 0.16

    score = max(0.0, min(score, 1.0))
    deep_rules = {name for name, _ in _DEEP_PATTERNS} | {"compound"}
    has_deep_signal = any(rule in deep_rules for rule in matched)

    if has_deep_signal or score >= 0.42:
        return RouteDecision(QueryRoute.DEEP, "Query needs multi-step or high-assurance handling.", score, matched)

    if fast_matches and len(tokens) <= 18:
        return RouteDecision(QueryRoute.FAST, "Short factual query fits the grounded fast path.", score, matched)

    if len(tokens) <= 8 and re.match(r"^\s*(who|what|when|where|define)\b", normalized, re.I):
        matched.append("short_wh")
        return RouteDecision(QueryRoute.FAST, "Short wh-question fits the grounded fast path.", score, matched)

    return RouteDecision(QueryRoute.DEEP, "Defaulting to QueryMind for safer synthesis.", max(score, 0.43), matched)
