from __future__ import annotations

import argparse
import json
import hashlib
import platform
from dataclasses import asdict
from datetime import datetime, timezone
import re
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.answer import Answerer, PROMPT_VERSION
from rag.index import RAGIndex
from rag.embeddings import Embeddings
from rag.local_models import LocalReranker
from scripts.validate_holdout import validate as validate_holdout


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


def model_metadata(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Local model directory does not exist: {root}")
    digest = hashlib.sha256()
    files = [file for file in sorted(root.rglob('*')) if file.is_file()]
    for file in files:
        digest.update(str(file.relative_to(root)).replace('\\', '/').encode('utf-8'))
        with file.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    return {'name': root.name, 'files': len(files), 'bytes': sum(file.stat().st_size for file in files), 'sha256': digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG retrieval and abstention evaluation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout", action="store_true", help="Evaluate a private, human-reviewed holdout instead of the 50-question development set")
    parser.add_argument("--development", type=Path, default=ROOT / "data" / "eval" / "questions.jsonl",
                        help="Development JSONL used to reject holdout question overlap")
    parser.add_argument("--local-embedding-model", help="Existing local sentence-transformers directory; no model download is attempted")
    parser.add_argument("--local-reranker-model", help="Existing local cross-encoder directory; no model download is attempted")
    args = parser.parse_args()
    questions_path = Path(args.questions)
    questions = load_questions(questions_path)
    holdout_validation: dict[str, object] | None = None
    if args.holdout:
        holdout_validation = validate_holdout(load_questions(args.development), questions)
        if holdout_validation["human_reviewed"] != holdout_validation["cases"]:
            holdout_validation["errors"].append("Every holdout case must be human-reviewed for this run")
            holdout_validation["valid"] = False
        if not holdout_validation["valid"]:
            raise SystemExit(f"Holdout validation failed: {holdout_validation['errors']}")
    elif len(questions) != 50:
        raise SystemExit(f"Expected 50 questions, found {len(questions)}")
    with tempfile.TemporaryDirectory() as temporary_directory:
        tracemalloc.start()
        embeddings = Embeddings(api_key="", local_model_path=args.local_embedding_model)
        index = RAGIndex(Path(temporary_directory) / "evaluation.sqlite3", embeddings=embeddings,
                         reranker=LocalReranker(args.local_reranker_model) if args.local_reranker_model else None)
        summary = index.index_directory(args.data)
        if summary.failed:
            raise SystemExit(f"Sample corpus had parsing failures: {summary.failed}")
        variants = [
            evaluate(index, questions, "vector", False),
            evaluate(index, questions, "keyword", False),
            evaluate(index, questions, "hybrid", False),
            evaluate(index, questions, "hybrid", True),
        ]
        python_peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        embedding_mode = index.embeddings.mode
        reranker_mode = index.status()['reranker_mode']
        index.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus_root = Path(args.data).resolve()
    file_hashes = {str(path.relative_to(corpus_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(corpus_root.rglob("*")) if path.is_file()}
    report = {
        "schema_version": 3,
        "run": {"time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "mode": "local-model" if args.local_embedding_model or args.local_reranker_model else "offline", "embedding": embedding_mode,
                "reranker": reranker_mode, "answer": "extractive", "top_k": 5,
                "dataset_kind": "human-reviewed-holdout" if args.holdout else "ai-authored-development",
                "lexical_profile": "enhanced-v1", "search_cache": True,
                "prompt_version": PROMPT_VERSION,
                "chunking": asdict(index.chunking), "pipeline_version": index.processing_version,
                "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
                "corpus_files_sha256": file_hashes,
                "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / "rag").glob("*.py")) + [Path(__file__).resolve()]},
                "api_calls": 0, "python_peak_bytes": python_peak_bytes,
                "local_models": {'embedding': model_metadata(args.local_embedding_model), 'reranker': model_metadata(args.local_reranker_model)}},
        "limitations": ["Citation presence and gold-file agreement are proxies, not semantic grounding.",
                        "Semantic citation precision and grounding are unmeasured (null).",
                        "Offline API cost is zero; local compute and human review costs are not measured.",
                        "Python allocation peak excludes native accelerator/runtime memory."] + (
                            ["The report stores no holdout question text, but the chosen local output path remains the user's privacy responsibility.",
                             "Human-reviewed labels are asserted by input metadata; reviewer identity and semantic judgment are not independently audited.",
                             "Holdout results are not a workplace outcome or a generalization claim."] if args.holdout else
                            ["Unanswerable set contains only three questions; no general abstention claim.",
                             "Local model quality needs an independently reviewed holdout."]),
        "holdout_validation": holdout_validation,
        "corpus": summary.indexed, "variants": variants,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EVALUATION PASSED: wrote {output} with {len(variants)} variants and {len(questions)} questions")


if __name__ == "__main__":
    main()
