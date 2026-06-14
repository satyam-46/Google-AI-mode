"""Arbitration agent node."""

from __future__ import annotations

from typing import Any

from core.chains.parsers import RetrievalEvidence


async def arbitrator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Mark low-confidence evidence for later conflict arbitration.

    Full claim-level conflict resolution belongs in the LangGraph phase. This
    node establishes the state contract with deterministic behavior today.
    """
    retrieval_results = state.get("retrieval_results") or []
    conflicts = [
        {
            "sub_question_id": item.get("sub_question_id", ""),
            "reason": "low_confidence_evidence",
            "confidence": item.get("confidence", 0.0),
        }
        for item in retrieval_results
        if isinstance(item, dict) and item.get("confidence", 0.0) < 0.5
    ]

    arbitration_results = [
        RetrievalEvidence(
            sub_question_id=conflict["sub_question_id"],
            answer_text="Evidence requires review before synthesis.",
            citations=[],
            confidence=conflict["confidence"],
        ).model_dump()
        for conflict in conflicts
    ]
    return {"conflicts_detected": conflicts, "arbitration_results": arbitration_results}
