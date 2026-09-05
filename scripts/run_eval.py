from __future__ import annotations

import argparse
import json
import hashlib
import platform
from datetime import datetime, timezone
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.answer import Answerer
from rag.index import RAGIndex
from rag.embeddings import Embeddings


def load_questions(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(index: RAGIndex, questions: list[dict[str, object]], mode: str, rerank: bool) -> dict[str, object]:
    answerer = Answerer(api_key="")
    retrieval_hits = citation_hits = citation_total = abstention_hits = abstention_total = grounded_hits = answerable_total = 0
    search_latencies: list[float] = []
    answer_latencies: list[float] = []
    total_latencies: list[float] = []
    rows: list[dict[str, object]] = []
    for question in questions:
        search_started = time.perf_counter()
        results = index.search(str(question["question"]), limit=5, mode=mode, rerank=rerank)
        search_latencies.append((time.perf_counter() - search_started) * 1000)
        expected = question.get("expected_document")
        answer_started = time.perf_counter()
        answer = answerer.answer(str(question["question"]), results)
        answer_latencies.append((time.perf_counter() - answer_started) * 1000)
        total_latencies.append((time.perf_counter() - search_started) * 1000)
        if expected:
            answerable_total += 1
            hit = any(Path(result.path).name == expected for result in results)
            retrieval_hits += int(hit)
            cited_numbers = [int(value) for value in re.findall(r"\[(\d+)\]", answer.text)]
            cited_sources = [source for source in answer.sources if source["number"] in cited_numbers]
            citation_hits += sum(int(Path(str(source["path"])).name == expected) for source in cited_sources)
            citation_total += len(cited_sources)
            grounded_hits += int(bool(cited_sources))
        else:
            abstention_total += 1
            abstention_hits += int(answer.status == "abstained")
            hit = answer.status == "abstained"
        rows.append({"id": question["id"], "hit": hit, "answer_status": answer.status, "top_result": Path(results[0].path).name if results else None})
    return {
        "mode": mode,
        "rerank": rerank,
        "questions": len(questions),
        "answerable_questions": answerable_total,
        "unanswerable_questions": abstention_total,
        "document_hit_at_5": retrieval_hits / max(1, answerable_total),
        "gold_document_citation_rate": citation_hits / max(1, citation_total),
        "citation_presence_rate": grounded_hits / max(1, answerable_total),
        "semantic_citation_precision": None,
        "grounded_answer_rate": None,
        "appropriate_abstention": abstention_hits / max(1, abstention_total),
        "p50_search_latency_ms": percentile(search_latencies, 0.5),
        "p95_search_latency_ms": percentile(search_latencies, 0.95),
        "p50_answer_latency_ms": percentile(answer_latencies, 0.5),
        "p95_answer_latency_ms": percentile(answer_latencies, 0.95),
        "p50_total_latency_ms": percentile(total_latencies, 0.5),
        "p95_total_latency_ms": percentile(total_latencies, 0.95),
        "estimated_cost_per_query_usd": 0.0,
        "cases": rows,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic local RAG retrieval and abstention evaluation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    questions = load_questions(Path(args.questions))
    if len(questions) != 50:
        raise SystemExit(f"Expected 50 questions, found {len(questions)}")
    with tempfile.TemporaryDirectory() as temporary_directory:
        index = RAGIndex(Path(temporary_directory) / "evaluation.sqlite3", embeddings=Embeddings(api_key=""))
        summary = index.index_directory(args.data)
        if summary.failed:
            raise SystemExit(f"Sample corpus had parsing failures: {summary.failed}")
        variants = [
            evaluate(index, questions, "vector", False),
            evaluate(index, questions, "keyword", False),
            evaluate(index, questions, "hybrid", False),
            evaluate(index, questions, "hybrid", True),
        ]
        index.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus_root = Path(args.data).resolve()
    file_hashes = {str(path.relative_to(corpus_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(corpus_root.rglob("*")) if path.is_file()}
    report = {
        "schema_version": 2,
        "run": {"time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "mode": "offline", "embedding": "local:hash-256", "answer": "extractive", "top_k": 5,
                "chunking": {"max_chars": 900, "overlap_chars": 160, "structure": True},
                "questions_sha256": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
                "corpus_files_sha256": file_hashes,
                "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / "rag").glob("*.py")) + [Path(__file__).resolve()]},
                "api_calls": 0},
        "limitations": ["Citation presence and gold-file agreement are proxies, not semantic grounding.",
                        "Semantic citation precision and grounding are unmeasured (null).",
                        "Unanswerable set contains only three questions; no general abstention claim.",
                        "Offline API cost is zero; local compute and human review costs are not measured."],
        "corpus": summary.indexed, "variants": variants,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EVALUATION PASSED: wrote {output} with {len(variants)} variants and {len(questions)} questions")


if __name__ == "__main__":
    main()
