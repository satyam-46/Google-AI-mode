"""Streamlit dashboard for QueryMind runtime observability."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from adk.callbacks import get_events
from adk.orchestrator import route_query
from core.memory.session_store import SessionStore
from observability.tracer import TraceStore

FLASH_COST_PER_1K_TOKENS = 0.00035
PRO_COST_PER_1K_TOKENS = 0.0035
DEFAULT_COST_PER_1K_TOKENS = 0.001


def summarize_session(record: dict[str, Any]) -> dict[str, Any]:
    """Create a compact, table-friendly summary for one saved session."""
    state = record.get("state") or {}
    final_answer = state.get("final_answer") or {}
    confidence_score = state.get("confidence_score") or {}
    citations = final_answer.get("citations") or []
    traces = state.get("agent_traces") or []
    updated_at = int(record.get("updated_at") or 0)

    confidence = final_answer.get("confidence")
    if confidence is None:
        confidence = confidence_score.get("score", 0.0)

    timeline = build_timeline(traces)
    total_tokens = int(state.get("total_tokens_used") or sum(item["tokens"] for item in timeline))
    latency_ms = int(state.get("total_latency_ms") or _timeline_latency_ms(timeline))

    return {
        "session_id": record.get("session_id", state.get("session_id", "")),
        "query": state.get("original_query", ""),
        "answer": final_answer.get("answer_text", ""),
        "confidence": round(float(confidence or 0.0), 3),
        "requires_human_review": bool(state.get("requires_human_review", False)),
        "citations": len(citations),
        "sub_questions": len(state.get("sub_questions") or []),
        "retrievals": len(state.get("retrieval_results") or []),
        "conflicts": len(state.get("conflicts_detected") or []),
        "trace_steps": len(traces),
        "latency_ms": latency_ms,
        "tokens": total_tokens,
        "estimated_cost_usd": round(estimate_trace_cost_usd(traces), 6),
        "cache_hits": len(state.get("cache_hits") or []),
        "cache_hit_rate": _cache_hit_rate(state),
        "updated_at": updated_at,
        "updated": _format_timestamp(updated_at),
    }


def build_dashboard_model(
    sessions: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    trace_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the aggregate model used by the Streamlit view and tests."""
    summaries = [summarize_session(record) for record in sessions]
    confidence_values = [item["confidence"] for item in summaries if item["confidence"] > 0]
    review_count = sum(1 for item in summaries if item["requires_human_review"])
    citation_count = sum(item["citations"] for item in summaries)
    latency_values = [item["latency_ms"] for item in summaries if item["latency_ms"] > 0]
    cache_hits = sum(item["cache_hits"] for item in summaries)
    retrievals = sum(item["retrievals"] for item in summaries)

    return {
        "sessions": summaries,
        "session_records": {record.get("session_id", ""): record for record in sessions},
        "events": sorted(events or [], key=lambda item: item.get("timestamp_ms", 0), reverse=True),
        "trace_records": trace_records or [],
        "metrics": {
            "total_sessions": len(summaries),
            "average_confidence": round(sum(confidence_values) / len(confidence_values), 3)
            if confidence_values
            else 0.0,
            "human_review": review_count,
            "citations": citation_count,
            "average_latency_ms": round(sum(latency_values) / len(latency_values)) if latency_values else 0,
            "cache_hit_rate": round(cache_hits / retrievals, 3) if retrievals else 0.0,
            "estimated_cost_usd": round(sum(item["estimated_cost_usd"] for item in summaries), 6),
        },
    }


async def load_dashboard_model(
    limit: int = 25,
    store: SessionStore | None = None,
    trace_store: TraceStore | None = None,
) -> dict[str, Any]:
    """Load recent sessions, ADK events, and persisted trace records."""
    session_store = store or SessionStore()
    traces = trace_store or TraceStore()
    sessions = await session_store.list_sessions(limit=limit)
    trace_records = await traces.list_records(limit=limit * 20)
    return build_dashboard_model(sessions, get_events(), trace_records)


def build_timeline(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize graph traces for Gantt-style rendering."""
    if not traces:
        return []
    origin = min(int(trace.get("start_ms") or 0) for trace in traces if trace.get("start_ms"))
    timeline = []
    for index, trace in enumerate(traces):
        start_ms = int(trace.get("start_ms") or origin)
        end_ms = int(trace.get("end_ms") or start_ms)
        details = trace.get("details") or {}
        status = details.get("status") or ("error" if details.get("errors") else "ok")
        tokens = int(trace.get("tokens_used") or 0)
        timeline.append(
            {
                "step": index + 1,
                "agent": trace.get("name", "unknown"),
                "status": status,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "offset_ms": start_ms - origin,
                "duration_ms": max(0, end_ms - start_ms),
                "tokens": tokens,
                "estimated_cost_usd": round(_estimate_agent_cost(trace), 6),
                "details": details,
            }
        )
    return timeline


def estimate_trace_cost_usd(traces: list[dict[str, Any]]) -> float:
    return sum(_estimate_agent_cost(trace) for trace in traces)


def build_session_trends(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return trend rows ordered by update time."""
    return [
        {
            "updated": item["updated"],
            "confidence": item["confidence"],
            "latency_ms": item["latency_ms"],
            "cache_hit_rate": item["cache_hit_rate"],
            "human_review": int(item["requires_human_review"]),
            "estimated_cost_usd": item["estimated_cost_usd"],
        }
        for item in sorted(sessions, key=lambda row: row["updated_at"])
    ]


def compare_replay(original: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """Compare a saved session summary with a replay result."""
    replay_answer = replay.get("answer", "")
    replay_citations = replay.get("citations") or []
    replay_confidence = float(replay.get("confidence") or 0.0)
    return {
        "original_confidence": original.get("confidence", 0.0),
        "replay_confidence": round(replay_confidence, 3),
        "confidence_delta": round(replay_confidence - float(original.get("confidence") or 0.0), 3),
        "original_citations": original.get("citations", 0),
        "replay_citations": len(replay_citations),
        "citation_delta": len(replay_citations) - int(original.get("citations") or 0),
        "original_answer_chars": len(original.get("answer") or ""),
        "replay_answer_chars": len(replay_answer),
        "route": replay.get("route", ""),
        "requires_human_review": bool(replay.get("requires_human_review", False)),
    }


async def replay_query(query: str, session_id: str) -> dict[str, Any]:
    """Rerun a past query through the orchestrator for dashboard comparison."""
    replay_session_id = f"{session_id}-replay-{int(time.time())}"
    return await route_query(query=query, session_id=replay_session_id)


def run_dashboard() -> None:
    """Render the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(page_title="QueryMind Observability", layout="wide")
    st.title("QueryMind Observability")

    with st.sidebar:
        st.header("Runtime")
        limit = st.slider("Recent sessions", min_value=5, max_value=100, value=25, step=5)
        auto_refresh = st.toggle("Auto-refresh", value=False)
        refresh = st.button("Refresh")

    model = asyncio.run(load_dashboard_model(limit=limit))
    metrics = model["metrics"]
    sessions = model["sessions"]

    cols = st.columns(6)
    cols[0].metric("Sessions", metrics["total_sessions"])
    cols[1].metric("Avg confidence", f"{metrics['average_confidence']:.2f}")
    cols[2].metric("Human review", metrics["human_review"])
    cols[3].metric("Cache hit rate", f"{metrics['cache_hit_rate']:.0%}")
    cols[4].metric("Avg latency", f"{metrics['average_latency_ms']} ms")
    cols[5].metric("Est. cost", f"${metrics['estimated_cost_usd']:.5f}")

    if refresh:
        st.rerun()
    if auto_refresh:
        time.sleep(3)
        st.rerun()

    if not sessions:
        st.info("No saved QueryMind sessions yet. Run the API or graph once to populate the dashboard.")
        return

    st.subheader("Session View")
    st.dataframe(_session_table_rows(sessions), use_container_width=True, hide_index=True)
    st.line_chart(build_session_trends(sessions), x="updated", y=["confidence", "cache_hit_rate", "human_review"])
    st.line_chart(build_session_trends(sessions), x="updated", y=["latency_ms"])

    selected_id = st.selectbox(
        "Inspect session",
        [item["session_id"] for item in sessions],
        format_func=lambda value: _session_label(value, sessions),
    )
    selected_record = next(record for record in model["sessions"] if record["session_id"] == selected_id)
    selected_state = model["session_records"].get(selected_id, {}).get("state", {})
    timeline = build_timeline(selected_state.get("agent_traces") or [])

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Answer")
        st.write(selected_record["answer"] or "No final answer recorded.")
    with right:
        st.subheader("Quality")
        st.metric("Confidence", f"{selected_record['confidence']:.2f}")
        st.metric("Trace steps", selected_record["trace_steps"])
        st.metric("Cache hits", selected_record["cache_hits"])
        st.metric("Conflicts", selected_record["conflicts"])

    st.subheader("Per-Query Timeline")
    if timeline:
        st.plotly_chart(build_timeline_figure(timeline), use_container_width=True)
    else:
        st.info("No agent trace timeline recorded for this session.")

    st.subheader("Token and Cost Breakdown")
    st.dataframe(
        [
            {
                "agent": item["agent"],
                "tokens": item["tokens"],
                "estimated_cost_usd": item["estimated_cost_usd"],
                "status": item["status"],
                "duration_ms": item["duration_ms"],
            }
            for item in timeline
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Conflict Log")
    st.json(
        {
            "conflicts_detected": selected_state.get("conflicts_detected") or [],
            "arbitration_results": selected_state.get("arbitration_results") or [],
        },
        expanded=False,
    )

    if st.button("Replay query"):
        if not selected_record["query"]:
            st.warning("This saved session has no original query to replay.")
        else:
            with st.spinner("Replaying query through the orchestrator..."):
                replay = asyncio.run(replay_query(selected_record["query"], selected_id))
            st.dataframe([compare_replay(selected_record, replay)], use_container_width=True, hide_index=True)
            with st.expander("Replay result"):
                st.json(replay, expanded=False)

    st.subheader("Graph State")
    st.json(selected_state, expanded=False)

    st.subheader("Recent ADK Events")
    st.dataframe(model["events"][:50], use_container_width=True, hide_index=True)

    st.subheader("Persisted Trace Records")
    st.dataframe(model["trace_records"][:100], use_container_width=True, hide_index=True)


def _session_table_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "updated": item["updated"],
            "session_id": item["session_id"],
            "query": item["query"],
            "confidence": item["confidence"],
            "review": item["requires_human_review"],
            "citations": item["citations"],
            "retrievals": item["retrievals"],
            "cache_hit_rate": item["cache_hit_rate"],
            "latency_ms": item["latency_ms"],
            "estimated_cost_usd": item["estimated_cost_usd"],
        }
        for item in sessions
    ]


def build_timeline_figure(timeline: list[dict[str, Any]]) -> Any:
    import plotly.graph_objects as go

    colors = {"ok": "#2ca02c", "partial": "#ffbf00", "failed": "#d62728", "error": "#d62728"}
    return go.Figure(
        data=[
            go.Bar(
                x=[item["duration_ms"] for item in timeline],
                y=[f"{item['step']}. {item['agent']}" for item in timeline],
                base=[item["offset_ms"] for item in timeline],
                orientation="h",
                marker_color=[colors.get(str(item["status"]), "#1f77b4") for item in timeline],
                customdata=[
                    [item["status"], item["tokens"], item["estimated_cost_usd"]]
                    for item in timeline
                ],
                hovertemplate=(
                    "status=%{customdata[0]}<br>"
                    "start=%{base} ms<br>"
                    "duration=%{x} ms<br>"
                    "tokens=%{customdata[1]}<br>"
                    "cost=$%{customdata[2]:.6f}<extra></extra>"
                ),
            )
        ]
    ).update_layout(
        xaxis_title="Milliseconds from query start",
        yaxis_title="Agent",
        showlegend=False,
        height=max(280, 56 * len(timeline)),
        margin={"l": 20, "r": 20, "t": 20, "b": 40},
    )


def _timeline_latency_ms(timeline: list[dict[str, Any]]) -> int:
    if not timeline:
        return 0
    starts = [int(item["start_ms"]) for item in timeline]
    ends = [int(item["end_ms"]) for item in timeline]
    return max(ends) - min(starts) if starts and ends else 0


def _estimate_agent_cost(trace: dict[str, Any]) -> float:
    tokens = int(trace.get("tokens_used") or 0)
    if tokens <= 0:
        return 0.0
    name = str(trace.get("name") or "")
    if name in {"planner", "synthesizer", "arbitrator"}:
        rate = PRO_COST_PER_1K_TOKENS
    elif name in {"retriever", "critic"}:
        rate = FLASH_COST_PER_1K_TOKENS
    else:
        rate = DEFAULT_COST_PER_1K_TOKENS
    return tokens / 1000 * rate


def _cache_hit_rate(state: dict[str, Any]) -> float:
    retrievals = len(state.get("retrieval_results") or [])
    if retrievals <= 0:
        return 1.0 if state.get("cache_hits") else 0.0
    return min(1.0, len(state.get("cache_hits") or []) / retrievals)


def _format_timestamp(timestamp: int) -> str:
    if not timestamp:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _session_label(session_id: str, sessions: list[dict[str, Any]]) -> str:
    for session in sessions:
        if session["session_id"] == session_id:
            query = session["query"][:80] or "untitled"
            return f"{query} ({session_id})"
    return session_id


if __name__ == "__main__":
    run_dashboard()
