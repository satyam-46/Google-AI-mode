import pytest

from agents.arbitrator import arbitrator_node
from agents.critic import critic_node
from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.synthesizer import synthesizer_node


@pytest.mark.asyncio
async def test_planner_node_returns_sub_questions():
    result = await planner_node({"original_query": "Where is England and how do accents differ?"})

    assert result["sub_questions"]
    assert result["sub_questions"][0]["id"] == "q1"
    assert 0 < result["complexity_score"] <= 1


@pytest.mark.asyncio
async def test_retriever_node_fans_out_sub_questions(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = await retriever_node(
        {
            "sub_questions": [
                {"id": "q1", "question": "What is Paris?"},
                {"id": "q2", "question": "What is Berlin?"},
            ]
        }
    )

    assert len(result["retrieval_results"]) == 2
    assert {item["sub_question_id"] for item in result["retrieval_results"]} == {"q1", "q2"}


@pytest.mark.asyncio
async def test_synthesizer_node_returns_final_answer():
    result = await synthesizer_node(
        {
            "original_query": "What is the capital of France?",
            "retrieval_results": [
                {
                    "sub_question_id": "q1",
                    "answer_text": "Paris is the capital of France.",
                    "citations": [
                        {
                            "source": "Paris",
                            "url": "https://example.com/paris",
                            "excerpt": "Paris is the capital.",
                        }
                    ],
                    "confidence": 0.8,
                }
            ],
        }
    )

    assert "Paris" in result["final_answer"]["answer_text"]
    assert result["final_answer"]["citations"]


@pytest.mark.asyncio
async def test_arbitrator_node_flags_low_confidence_evidence():
    result = await arbitrator_node(
        {
            "retrieval_results": [
                {"sub_question_id": "q1", "answer_text": "No evidence found.", "citations": [], "confidence": 0.0}
            ]
        }
    )

    assert result["conflicts_detected"][0]["reason"] == "low_confidence_evidence"
    assert result["arbitration_results"][0]["sub_question_id"] == "q1"


@pytest.mark.asyncio
async def test_critic_node_requires_review_for_low_evidence():
    result = await critic_node({"final_answer": {"answer_text": "No evidence found.", "citations": [], "confidence": 0.0}})

    assert result["requires_human_review"] is True
    assert "low_evidence" in result["confidence_score"]["flags"]
