from typing import Any, Dict
import asyncio


async def planner_chain(prompt: str, **kwargs) -> Dict[str, Any]:
    """Stub planner chain that returns a plan dict."""
    await asyncio.sleep(0)
    return {"plan": [{"step": "search", "query": prompt}]}


async def retriever_chain(query: str, top_k: int = 5) -> Dict[str, Any]:
    await asyncio.sleep(0)
    return {"documents": []}


async def synthesizer_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0)
    return {"answer": "stub"}
