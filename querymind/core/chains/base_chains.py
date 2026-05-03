"""LCEL chains for the Phase 1 QueryMind system."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from core.chains.parsers import (
    CitedAnswer,
    CitedAnswerParser,
    ConfidenceParser,
    ConfidenceScore,
    RetrievalEvidence,
    RetrievalEvidenceParser,
    SubQuestionList,
    SubQuestionParser,
)
from core.chains.prompts import CRITIC_PROMPT, PLANNER_PROMPT, RETRIEVER_PROMPT, SYNTHESIZER_PROMPT
from core.tools.web_search import SearchResult, web_search


def _chat_text(value: Any) -> str:
    if isinstance(value, ChatPromptValue):
        return "\n".join(message.content for message in value.messages)
    if isinstance(value, BaseMessage):
        return str(value.content)
    return str(value)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True)


def _extract_after(label: str, text: str) -> str:
    pattern = rf"{re.escape(label)}\s*(.*)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _split_query(query: str) -> list[str]:
    parts = [part.strip(" ?.") for part in re.split(r"\band\b|,|;|\n", query) if part.strip(" ?.")]
    if len(parts) <= 1:
        return [query.strip(" ?.")]
    return parts[:5]


def _fake_planner_llm(prompt: Any) -> str:
    text = _chat_text(prompt)
    query = _extract_after("User query:", text)
    questions = []
    for index, part in enumerate(_split_query(query), start=1):
        question = part if part.endswith("?") else f"{part}?"
        questions.append(
            {
                "id": f"q{index}",
                "question": question,
                "search_query": part,
                "reasoning": "This isolates one searchable part of the user query.",
            }
        )
    return _json({"sub_questions": questions})


def _fake_retriever_llm(prompt: Any) -> str:
    text = _chat_text(prompt)
    sub_question_id = _extract_field(text, "Sub-question id")
    documents = _extract_after("Evidence documents:", text)
    if not documents or documents == "[]":
        return _json(
            {
                "sub_question_id": sub_question_id,
                "answer_text": "No evidence found.",
                "citations": [],
                "confidence": 0.0,
            }
        )

    try:
        parsed_docs = json.loads(documents)
    except json.JSONDecodeError:
        parsed_docs = []

    citations = []
    snippets = []
    for doc in parsed_docs[:3]:
        title = doc.get("title") or doc.get("url") or "Source"
        content = doc.get("content") or ""
        snippets.append(content)
        citations.append({"source": title, "url": doc.get("url") or "", "excerpt": content[:240]})

    return _json(
        {
            "sub_question_id": sub_question_id,
            "answer_text": " ".join(snippets).strip() or "No evidence found.",
            "citations": citations,
            "confidence": 0.7 if citations else 0.0,
        }
    )


def _fake_synthesizer_llm(prompt: Any) -> str:
    text = _chat_text(prompt)
    evidence_text = _extract_after("Evidence:", text)
    try:
        evidence = json.loads(evidence_text)
    except json.JSONDecodeError:
        evidence = []

    answer_parts: list[str] = []
    citations: list[dict[str, str]] = []
    confidences: list[float] = []
    for item in evidence:
        if item.get("answer_text") and item["answer_text"] != "No evidence found.":
            answer_parts.append(item["answer_text"])
        citations.extend(item.get("citations") or [])
        confidences.append(float(item.get("confidence") or 0.0))

    answer_text = " ".join(answer_parts).strip() or "No evidence found."
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return _json({"answer_text": answer_text, "citations": citations, "confidence": confidence})


def _fake_critic_llm(prompt: Any) -> str:
    text = _chat_text(prompt)
    answer = _extract_after("Answer to check:", text)
    sensitive = [flag for flag in ("medical", "legal", "financial") if flag in answer.lower()]
    unsupported = "No evidence found" in answer
    score = 0.35 if unsupported else 0.8
    return _json(
        {
            "score": score,
            "reasoning": "Offline critic fallback based on evidence availability and sensitive-topic flags.",
            "flags": sensitive + (["low_evidence"] if unsupported else []),
        }
    )


def _extract_field(text: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _model(model_name: str, fake_llm: Any) -> Runnable[Any, str]:
    if not os.getenv("GOOGLE_API_KEY"):
        return RunnableLambda(fake_llm)

    primary = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    fallback = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return primary.with_fallbacks([fallback])


async def _prepare_retriever_input(inputs: dict[str, Any]) -> dict[str, Any]:
    question = str(inputs.get("question") or inputs.get("query") or "")
    sub_question_id = str(inputs.get("sub_question_id") or inputs.get("id") or "")
    provided_docs = inputs.get("documents")

    if provided_docs is None:
        results = await web_search(question, top_k=int(inputs.get("top_k", 5)))
        provided_docs = [result.model_dump() for result in results]
    elif all(isinstance(item, SearchResult) for item in provided_docs):
        provided_docs = [item.model_dump() for item in provided_docs]

    return {
        "sub_question_id": sub_question_id,
        "question": question,
        "documents": json.dumps(provided_docs, ensure_ascii=True),
    }


def _prepare_synthesizer_input(inputs: dict[str, Any]) -> dict[str, Any]:
    evidence = inputs.get("evidence") or inputs.get("retrieval_results") or []
    normalized = [item.model_dump() if hasattr(item, "model_dump") else item for item in evidence]
    return {"query": inputs.get("query") or inputs.get("original_query") or "", "evidence": _json(normalized)}


def _prepare_critic_input(inputs: dict[str, Any] | CitedAnswer) -> dict[str, Any]:
    if isinstance(inputs, CitedAnswer):
        return {"answer": inputs.model_dump_json()}
    answer = inputs.get("answer") or inputs.get("final_answer") or inputs
    if hasattr(answer, "model_dump_json"):
        answer = answer.model_dump_json()
    return {"answer": str(answer)}


planner_chain: Runnable[dict[str, Any], SubQuestionList] = (
    PLANNER_PROMPT | _model("gemini-1.5-pro", _fake_planner_llm) | StrOutputParser() | SubQuestionParser
)

retriever_chain: Runnable[dict[str, Any], RetrievalEvidence] = (
    RunnableLambda(_prepare_retriever_input)
    | RETRIEVER_PROMPT
    | _model("gemini-1.5-flash", _fake_retriever_llm)
    | StrOutputParser()
    | RetrievalEvidenceParser
)

synthesizer_chain: Runnable[dict[str, Any], CitedAnswer] = (
    RunnableLambda(_prepare_synthesizer_input)
    | SYNTHESIZER_PROMPT
    | _model("gemini-1.5-pro", _fake_synthesizer_llm)
    | StrOutputParser()
    | CitedAnswerParser
)

critic_chain: Runnable[dict[str, Any] | CitedAnswer, ConfidenceScore] = (
    RunnableLambda(_prepare_critic_input)
    | CRITIC_PROMPT
    | _model("gemini-1.5-flash", _fake_critic_llm)
    | StrOutputParser()
    | ConfidenceParser
)


__all__ = ["critic_chain", "planner_chain", "retriever_chain", "synthesizer_chain"]
