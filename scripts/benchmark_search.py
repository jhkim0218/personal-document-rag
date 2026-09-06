from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embeddings import Embeddings
from rag.index import RAGIndex
from scripts.run_eval import load_questions, percentile


def signature(results):
    return [(row.chunk_id, row.score) for row in results]


def scale_case(size, repetitions):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "documents"
        source.mkdir()
        for number in range(size):
            (source / f"project-{number:05}.txt").write_text(
                f"프로젝트 PROJECT_{number:05} 배포 일정은 9월 {number%28+1}일입니다. "
                f"ERR_CONN_{number:05} 연결 오류는 운영팀에 문의하세요.\n" * 3, encoding="utf-8")
        database = root / "index.db"
        index = RAGIndex(database, Embeddings(api_key=""))
        tracemalloc.start()
        started = time.perf_counter()
        indexed = index.index_directory(source)
        index_seconds = time.perf_counter() - started
        index_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        if indexed.failed or indexed.indexed != size:
            raise AssertionError("Benchmark indexing incomplete")
        index.close()
        queries = [f"PROJECT_{i%size:05} 배포 일정" for i in range(repetitions)]
        reference = {}
        measurements = []
        for mode in ("keyword", "hybrid"):
            for cached in (False, True):
                tracemalloc.start()
                index = RAGIndex(database, Embeddings(api_key=""))
                latencies = []
                try:
                    for query in queries:
                        started = time.perf_counter()
                        rows = index.search(query, mode=mode, use_cache=cached)
                        latencies.append((time.perf_counter()-started)*1000)
                        key = (mode, query)
                        if cached:
                            if signature(rows) != reference[key]:
                                raise AssertionError("Cached ranking/score differs from uncached control")
                        else:
                            reference[key] = signature(rows)
                    measurements.append({"mode": mode, "cached": cached, "queries": len(queries),
                        "first_query_ms": round(latencies[0], 3), "p50_ms": percentile(latencies, .5),
                        "p95_ms": percentile(latencies, .95), "python_peak_bytes": tracemalloc.get_traced_memory()[1],
                        "cache_builds": index.cache_builds})
                finally:
                    index.close()
                    tracemalloc.stop()
        return {"documents": size, "pipeline_version": index.processing_version, "index_seconds": round(index_seconds, 3), "index_python_peak_bytes": index_peak,
                "cache_score_equivalence": True, "measurements": measurements}


def quality_cases():
    fixture_path = ROOT / "data/eval/retrieval_cases.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for relative, content in fixture["documents"].items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        index = RAGIndex(root / "index.db", Embeddings(api_key=""))
        try:
            index.index_directory(root)
            rows = []
            for case in fixture["queries"]:
                result = {**case}
                for lexical in ("legacy", "enhanced"):
                    found = index.search(case["query"], mode="keyword", rerank=False, lexical=lexical)
                    result[lexical] = Path(found[0].path).relative_to(root).as_posix() if found else None
                    result[lexical + "_hit_at_1"] = result[lexical] == case["expected"]
                rows.append(result)
        finally:
            index.close()
    groups = []
    for kind in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == kind]
        groups.append({"type": kind, "count": len(group), **{lexical: sum(r[lexical+"_hit_at_1"] for r in group)/len(group) for lexical in ("legacy", "enhanced")}})
    return {"provenance": fixture["provenance"], "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(), "by_type_hit_at_1": groups, "cases": rows}


def sample_regression():
    questions = load_questions(ROOT / "data/eval/questions.jsonl")
    with tempfile.TemporaryDirectory() as temporary:
        index = RAGIndex(Path(temporary)/"index.db", Embeddings(api_key=""))
        variants = []
        try:
            index.index_directory(ROOT / "data/sample")
            for mode, rerank in (("vector", False), ("keyword", False), ("hybrid", False), ("hybrid", True)):
                for lexical in ("legacy", "enhanced"):
                    cases = []
                    for question in questions:
                        if not question.get("expected_document"):
                            continue
                        results = index.search(question["question"], mode=mode, rerank=rerank, lexical=lexical)
                        control = index.search(question["question"], mode=mode, rerank=rerank, lexical=lexical, use_cache=False)
                        if signature(results) != signature(control):
                            raise AssertionError("Sample cached results differ")
                        cases.append({"id": question["id"], "hit": any(Path(r.path).name == question["expected_document"] for r in results)})
                    variants.append({"mode": mode, "rerank": rerank, "lexical": lexical, "document_hit_at_5": sum(r["hit"] for r in cases)/len(cases), "cases": cases})
        finally:
            index.close()
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 200, 1000])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", default="results/search_benchmark.json")
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repetitions < 2:
        parser.error("positive sizes and at least two repetitions required")
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "embedding": "local:hash-256", "api_calls": 0,
              "lexical_profile": "enhanced-v1",
              "code_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"rag/index.py", ROOT/"rag/text.py", Path(__file__).resolve()]},
              "limitations": ["Synthetic short-document scale test, not workplace latency.", "Tracemalloc measures Python allocations, not total process RSS; instrumentation adds overhead.", "First query is cold; p50/p95 include first query. Uncached runs precede cached runs.", "Development labels are AI-authored, not human-reviewed holdout evidence."],
              "scale": [], "development_quality": quality_cases(), "sample_regression": sample_regression()}
    for size in args.sizes:
        report["scale"].append(scale_case(size, args.repetitions))
        print(f"Measured {size} synthetic documents", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SEARCH BENCHMARK PASSED: score equivalence verified; measurements written")


if __name__ == "__main__":
    main()
