from typing import Dict, Any


async def synthesizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub synthesizer that composes a simple answer."""
    final = {"answer_text": "This is a synthesized stub answer.", "citations": []}
    return {"final_answer": final}
