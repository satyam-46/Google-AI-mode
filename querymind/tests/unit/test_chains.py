import pytest

from core.chains.base_chains import arbitrator_chain, critic_chain, planner_chain, retriever_chain, synthesizer_chain
from core.chains.parsers import (
    ArbitrationResult,
    CitedAnswer,
    ConfidenceScore,
    FixedCitedAnswerParser,
    RetrievalEvidence,
    SubQuestionList,
)


@pytest.mark.asyncio
async def test_planner_chain_returns_structured_sub_questions():
    result = await planner_chain.ainvoke({"query": "Compare LangChain and LangGraph"})

    assert isinstance(result, SubQuestionList)
    assert result.sub_questions
    assert result.sub_questions[0].search_query


@pytest.mark.asyncio
async def test_retriever_chain_extracts_evidence_from_documents():
    result = await retriever_chain.ainvoke(
        {
            "sub_question_id": "q1",
            "question": "What is Paris?",
            "documents": [
                {
                    "title": "Paris",
                    "url": "https://example.com/paris",
                    "content": "Paris is the capital of France.",
                    "relevance_score": 0.9,
                }
            ],
        }
    )

    assert isinstance(result, RetrievalEvidence)
    assert "Paris" in result.answer_text
    assert result.citations[0].url == "https://example.com/paris"


@pytest.mark.asyncio
async def test_synthesizer_chain_composes_cited_answer():
    evidence = [
        RetrievalEvidence(
            sub_question_id="q1",
            answer_text="Paris is the capital of France.",
            citations=[{"source": "Paris", "url": "https://example.com/paris", "excerpt": "Paris is the capital."}],
            confidence=0.8,
        )
    ]

    result = await synthesizer_chain.ainvoke({"query": "What is the capital of France?", "evidence": evidence})

    assert isinstance(result, CitedAnswer)
    assert "Paris" in result.answer_text
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_critic_chain_flags_low_evidence():
    result = await critic_chain.ainvoke({"answer": "No evidence found."})

    assert isinstance(result, ConfidenceScore)
    assert result.score < 0.6
    assert "low_evidence" in result.flags


@pytest.mark.asyncio
async def test_arbitrator_chain_resolves_conflict():
    result = await arbitrator_chain.ainvoke(
        {
            "conflict": {
                "entity": "Python",
                "claim_a": "Python 2 was released in 2000.",
                "source_a": "https://example.com/a",
                "claim_b": "Python 2 was released in 1991.",
                "source_b": "https://example.com/b",
            }
        }
    )

    assert isinstance(result, ArbitrationResult)
    assert result.winning_claim
    assert result.reasoning


@pytest.mark.asyncio
async def test_all_phase1_chains_stream_structured_outputs():
    chains_with_inputs = [
        (planner_chain, {"query": "Compare LangChain and LangGraph"}),
        (
            retriever_chain,
            {
                "sub_question_id": "q1",
                "question": "What is Paris?",
                "documents": [
                    {
                        "title": "Paris",
                        "url": "https://example.com/paris",
                        "content": "Paris is the capital of France.",
                        "relevance_score": 0.9,
                    }
                ],
            },
        ),
        (
            synthesizer_chain,
            {
                "query": "What is the capital of France?",
                "evidence": [
                    RetrievalEvidence(
                        sub_question_id="q1",
                        answer_text="Paris is the capital of France.",
                        citations=[{"source": "Paris", "url": "https://example.com/paris"}],
                        confidence=0.8,
                    )
                ],
            },
        ),
        (arbitrator_chain, {"conflict": {"entity": "Python", "claim_a": "Python 2 released in 2000."}}),
        (critic_chain, {"answer": "Paris is supported by evidence."}),
    ]

    for chain, payload in chains_with_inputs:
        chunks = [chunk async for chunk in chain.astream(payload)]
        assert chunks


def test_output_fixing_parser_repairs_malformed_json():
    malformed = """
    ```json
    {'answer_text': 'Paris is the capital.', 'citations': [], 'confidence': 0.82,}
    ```
    """

    result = FixedCitedAnswerParser.parse(malformed)

    assert isinstance(result, CitedAnswer)
    assert result.confidence == 0.82
