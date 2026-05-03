"""Pydantic models and LangChain output parsers for QueryMind."""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str = Field(..., description="Stable sub-question identifier.")
    question: str = Field(..., description="Natural language sub-question.")
    search_query: str = Field(..., description="Search-engine optimized query.")
    reasoning: str = Field(..., description="Why this sub-question is useful.")


class SubQuestionList(BaseModel):
    sub_questions: list[SubQuestion]


class Citation(BaseModel):
    source: str
    url: str = ""
    excerpt: str = ""


class CitedAnswer(BaseModel):
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ConfidenceScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    flags: list[str] = Field(default_factory=list)


class RetrievalEvidence(BaseModel):
    sub_question_id: str = ""
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


SubQuestionParser = PydanticOutputParser(pydantic_object=SubQuestionList)
CitedAnswerParser = PydanticOutputParser(pydantic_object=CitedAnswer)
ConfidenceParser = PydanticOutputParser(pydantic_object=ConfidenceScore)
RetrievalEvidenceParser = PydanticOutputParser(pydantic_object=RetrievalEvidence)


__all__ = [
    "Citation",
    "CitedAnswer",
    "CitedAnswerParser",
    "ConfidenceParser",
    "ConfidenceScore",
    "RetrievalEvidence",
    "RetrievalEvidenceParser",
    "SubQuestion",
    "SubQuestionList",
    "SubQuestionParser",
]
