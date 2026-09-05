from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict

from .index import SearchResult
from .text import split_sentences


@dataclass(frozen=True)
class Answer:
    status: str
    text: str
    sources: list[dict[str, object]]
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Answerer:
    """Evidence-only answer generation with an offline extractive fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("RAG_MODEL", "gpt-5.2")

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        supported = [result for result in results if result.overlap > 0 or result.keyword_score > 0]
        if not supported:
            return Answer(
                status="abstained",
                text="색인된 문서에서 질문을 뒷받침할 근거를 찾지 못했습니다. 다른 표현으로 검색하거나 관련 문서를 추가해 주세요.",
                sources=[],
            )
        conflict = detect_possible_conflict(question, supported)
        evidence_results = supported if self.api_key else supported[: 2 if conflict else 1]
        sources = [
            {"number": number, "chunk_id": result.chunk_id, "title": result.title, "path": result.path, "location": result.location, "excerpt": result.text[:500]}
            for number, result in enumerate(evidence_results, start=1)
        ]
        if self.api_key:
            try:
                text = self._openai_answer(question, sources)
                citation_error = validate_citations(text, len(sources))
                if citation_error:
                    return Answer(status="fallback", text=self._extractive_answer(supported), sources=sources, error=citation_error)
                return Answer(status="conflict" if conflict else "answered", text=_with_conflict_notice(text, conflict), sources=sources)
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as error:
                return Answer(status="conflict" if conflict else "fallback", text=_with_conflict_notice(self._extractive_answer(evidence_results), conflict), sources=sources, error=str(error))
        return Answer(status="conflict" if conflict else "fallback", text=_with_conflict_notice(self._extractive_answer(evidence_results), conflict), sources=sources)

    def _extractive_answer(self, results: list[SearchResult]) -> str:
        lines: list[str] = []
        for number, result in enumerate(results[:3], start=1):
            sentence = next(iter(split_sentences(result.text)), result.text).strip()
            lines.append(f"- {sentence} [{number}]")
        return "문서에서 확인된 관련 내용입니다:\n" + "\n".join(lines)

    def _openai_answer(self, question: str, sources: list[dict[str, object]]) -> str:
        evidence = "\n\n".join(
            f"[{source['number']}] {source['title']} — {source['location']}\n{source['excerpt']}" for source in sources
        )
        instructions = (
            "You answer questions only from the supplied evidence. Write Korean unless the user asks otherwise. "
            "Every factual sentence must end with one or more citation numbers such as [1]. "
            "If the evidence is insufficient or conflicting, say so plainly instead of guessing."
        )
        payload = json.dumps(
            {
                "model": self.model,
                "instructions": instructions,
                "input": f"Question:\n{question}\n\nEvidence:\n{evidence}",
                "store": False,
                "text": {"verbosity": "low"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
        texts = [
            content["text"]
            for item in body.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text" and content.get("text")
        ]
        if not texts:
            raise ValueError("The model response did not contain output text")
        return "\n".join(texts).strip()


def validate_citations(text: str, source_count: int) -> str | None:
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", text)]
    if not citations:
        return "The generated answer did not include any citations"
    if any(number < 1 or number > source_count for number in citations):
        return "The generated answer included an invalid citation number"
    return None


def detect_possible_conflict(question: str, results: list[SearchResult]) -> str | None:
    """Flag competing date/number claims from distinct documents for manual source review."""
    numeric_question_markers = ("when", "date", "schedule", "how many", "how much", "time", "cost", "언제", "날짜", "일정", "몇", "얼마", "시간", "비용", "수치")
    if not any(marker in question.lower() for marker in numeric_question_markers):
        return None
    values_by_document: dict[str, set[str]] = {}
    pattern = re.compile(r"\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b|\b\d+(?:\.\d+)?(?:%|ms|초|분|일|개)?\b")
    for result in results:
        values_by_document.setdefault(result.path, set()).update(pattern.findall(result.text))
    distinct_values = set().union(*values_by_document.values()) if values_by_document else set()
    if len(values_by_document) >= 2 and len(distinct_values) >= 2:
        return "서로 다른 문서에 서로 다른 날짜 또는 수치가 있습니다. 아래 인용 원문을 확인해 어느 값이 현재 질문에 적용되는지 판단하세요."
    return None


def _with_conflict_notice(text: str, conflict: str | None) -> str:
    return f"{text}\n\n주의: {conflict}" if conflict else text
