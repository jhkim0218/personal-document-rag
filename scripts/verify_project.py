from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required = [
        "app.py",
        "requirements.txt",
        "README.md",
        "PERSONAL_DOCUMENT_RAG_DESIGN.md",
        "IMPLEMENTATION_NOTES.md",
        "rag/documents.py",
        "rag/index.py",
        "rag/answer.py",
        "data/eval/questions.jsonl",
    ]
    for relative_path in required:
        require((ROOT / relative_path).is_file(), f"Missing required artifact: {relative_path}")
    documents = list((ROOT / "data/sample").glob("*"))
    require(len(documents) >= 20, "Expected at least 20 sample documents")
    questions = [json.loads(line) for line in (ROOT / "data/eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    require(len(questions) == 50, "Expected exactly 50 evaluation questions")
    require(any(not question["answerable"] for question in questions), "Expected unanswerable evaluation questions")
    design = (ROOT / "PERSONAL_DOCUMENT_RAG_DESIGN.md").read_text(encoding="utf-8")
    implementation = (ROOT / "IMPLEMENTATION_NOTES.md").read_text(encoding="utf-8")
    for phrase in ("업무 효율", "경력서", "증분 색인", "답변 보류"):
        require(phrase in design, f"Design is missing: {phrase}")
    for phrase in ("무엇을", "왜", "어떻게", "검증"):
        require(phrase in implementation, f"Implementation notes are missing: {phrase}")
    print("PROJECT VERIFICATION PASSED")


if __name__ == "__main__":
    main()
