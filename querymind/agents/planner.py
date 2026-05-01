from typing import Dict, Any
import asyncio


async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub planner node: returns a simple sub_questions list and complexity score."""
    # Minimal deterministic behaviour for scaffold
    query = state.get("original_query", "")
    sub_questions = [{"id": "q1", "question": query}]
    return {"sub_questions": sub_questions, "complexity_score": 0.2}
