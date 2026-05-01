from typing import Dict, Any


async def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub retriever that returns an empty retrieval result.

    Real implementation should call Tavily + document store.
    """
    return {"retrieval_results": []}
