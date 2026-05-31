"""ADK tools exposed by QueryMind."""

from adk.tools.querymind_tool import run_querymind

try:
    from adk.tools.querymind_tool import querymind_tool
except Exception:  # pragma: no cover - ADK import failures should not break local tests.
    querymind_tool = None

__all__ = ["querymind_tool", "run_querymind"]

