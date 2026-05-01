from typing import Dict, Any


async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub critic that returns a high confidence score."""
    return {"confidence_score": {"score": 0.95, "flags": []}, "requires_human_review": False}
