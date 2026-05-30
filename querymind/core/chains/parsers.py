"""Pydantic models and LangChain output parsers for QueryMind."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Generic, TypeVar

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

T = TypeVar("T")


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


class ArbitrationResult(BaseModel):
    entity: str = "unknown"
    winning_claim: str
    winning_source: str = ""
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class OutputFixingParser(BaseOutputParser[T], Generic[T]):
    """Small local fixing parser for malformed JSON in LangChain v1.

    The legacy LangChain `OutputFixingParser` is not available in the installed
    package layout. This wrapper preserves the Phase 1 contract: strict parse
    first, deterministic JSON repair second, then Pydantic validation.
    """

    parser: PydanticOutputParser

    @property
    def _type(self) -> str:
        return "querymind_output_fixing_parser"

    def parse(self, text: str) -> T:
        try:
            return self.parser.parse(text)
        except Exception:
            return self.parser.parse(_repair_json(text))


def _repair_json(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        parsed: Any = ast.literal_eval(cleaned)
        return json.dumps(parsed, ensure_ascii=True)
    except Exception:
        pass

    return re.sub(r"(?<!\")\b([A-Za-z_][A-Za-z0-9_]*)\b\s*:", r'"\1":', cleaned)


SubQuestionParser = PydanticOutputParser(pydantic_object=SubQuestionList)
CitedAnswerParser = PydanticOutputParser(pydantic_object=CitedAnswer)
ConfidenceParser = PydanticOutputParser(pydantic_object=ConfidenceScore)
RetrievalEvidenceParser = PydanticOutputParser(pydantic_object=RetrievalEvidence)
ArbitrationResultParser = PydanticOutputParser(pydantic_object=ArbitrationResult)

FixedSubQuestionParser = OutputFixingParser[SubQuestionList](parser=SubQuestionParser)
FixedCitedAnswerParser = OutputFixingParser[CitedAnswer](parser=CitedAnswerParser)
FixedConfidenceParser = OutputFixingParser[ConfidenceScore](parser=ConfidenceParser)
FixedRetrievalEvidenceParser = OutputFixingParser[RetrievalEvidence](parser=RetrievalEvidenceParser)
FixedArbitrationResultParser = OutputFixingParser[ArbitrationResult](parser=ArbitrationResultParser)


__all__ = [
    "ArbitrationResult",
    "ArbitrationResultParser",
    "Citation",
    "CitedAnswer",
    "CitedAnswerParser",
    "ConfidenceParser",
    "ConfidenceScore",
    "FixedArbitrationResultParser",
    "FixedCitedAnswerParser",
    "FixedConfidenceParser",
    "FixedRetrievalEvidenceParser",
    "FixedSubQuestionParser",
    "OutputFixingParser",
    "RetrievalEvidence",
    "RetrievalEvidenceParser",
    "SubQuestion",
    "SubQuestionList",
    "SubQuestionParser",
]
