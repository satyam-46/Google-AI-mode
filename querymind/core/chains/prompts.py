"""Chat prompt templates for the Phase 1 LangChain layer."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .parsers import CitedAnswerParser, ConfidenceParser, RetrievalEvidenceParser, SubQuestionParser


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decompose user questions into focused web-search sub-questions. "
            "Return only JSON that matches the schema.\n{format_instructions}",
        ),
        ("human", "User query: {query}"),
    ]
).partial(format_instructions=SubQuestionParser.get_format_instructions())

RETRIEVER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract concise evidence from retrieved documents. "
            "Return only JSON that matches the schema.\n{format_instructions}",
        ),
        (
            "human",
            "Sub-question id: {sub_question_id}\nSub-question: {question}\nEvidence documents:\n{documents}",
        ),
    ]
).partial(format_instructions=RetrievalEvidenceParser.get_format_instructions())

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You synthesize grounded answers from evidence. Cite sources from the provided evidence. "
            "Return only JSON that matches the schema.\n{format_instructions}",
        ),
        ("human", "Original query: {query}\nEvidence:\n{evidence}"),
    ]
).partial(format_instructions=CitedAnswerParser.get_format_instructions())

ARBITRATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You resolve conflicting factual claims by weighing source authority, recency, and corroboration.",
        ),
        ("human", "Conflict to resolve:\n{conflict}"),
    ]
)

CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check an answer for unsupported claims and sensitive topics. "
            "Return only JSON that matches the schema.\n{format_instructions}",
        ),
        ("human", "Answer to check:\n{answer}"),
    ]
).partial(format_instructions=ConfidenceParser.get_format_instructions())


__all__ = [
    "ARBITRATOR_PROMPT",
    "CRITIC_PROMPT",
    "PLANNER_PROMPT",
    "RETRIEVER_PROMPT",
    "SYNTHESIZER_PROMPT",
]
