from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.documents import Chunking, parse_document
from rag.embeddings import Embeddings
from rag.index import RAGIndex
from scripts.run_eval import percentile


def evidence_hits(gold, results, root):
    # Literal complete-span recall, not entailment or generated-answer accuracy.
    return [any(Path(row.path).relative_to(root).as_posix() == item['document']
                and item['text'] in row.text for row in results) for item in gold]


def validate_labels(fixture, root):
    if not fixture['queries']:
        raise ValueError('At least one evaluation query required')
    ids = set()
    for case in fixture['queries']:
        if case['id'] in ids or not case['gold'] or not case['question'].strip():
            raise ValueError('Unique case IDs and nonempty gold evidence required')
        ids.add(case['id'])
        for gold in case['gold']:
            if gold['document'] not in fixture['documents'] or not gold['text'].strip():
                raise ValueError('Gold document missing or evidence empty')
            parsed = parse_document(root/gold['document'])
            if not any(s.location == gold['location'] and gold['text'] in s.text for s in parsed.sections):
                raise ValueError(f"Gold evidence not found at source location: {case['id']}")


def compare(fixture_path, max_chars=900, overlap_chars=160):
    configs = [Chunking(strategy, max_chars, overlap_chars) for strategy in ('fixed', 'structured')]
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    variants = []
    with tempfile.TemporaryDirectory() as temporary, patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline experiment attempted network')):
        root = Path(temporary)/'corpus'
        root.mkdir()
        for name, text in fixture['documents'].items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()) or target.suffix != '.md':
                raise ValueError('Fixture documents must be Markdown paths inside corpus')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding='utf-8')
        validate_labels(fixture, root)
        for config in configs:
            index = RAGIndex(Path(temporary)/f'{config.strategy}.db', Embeddings(api_key=''), chunking=config)
            try:
                started = time.perf_counter()
                summary = index.index_directory(root)
                index_ms = (time.perf_counter()-started)*1000
                if summary.failed or summary.indexed != len(fixture['documents']):
                    raise AssertionError('Incomplete experiment index')
                cases, latencies = [], []
                for case in fixture['queries']:
                    started = time.perf_counter()
                    rows = index.search(case['question'], limit=5, mode='hybrid', rerank=False)
                    latencies.append((time.perf_counter()-started)*1000)
                    hits = evidence_hits(case['gold'], rows, root)
                    # Separate evidence lost at chunk boundaries from search ranking failures.
                    available = []
                    for gold in case['gold']:
                        available.append(any(gold['text'] in row[0] for row in index.connection.execute(
                            'SELECT text FROM chunks WHERE path=?', (str((root/gold['document']).resolve()),))))
                    cases.append({'id': case['id'], 'question': case['question'], 'type': case['type'],
                        'gold': case['gold'], 'hits': hits, 'available_in_single_chunk': available,
                        'recall': sum(hits)/len(hits), 'all_evidence_found': all(hits),
                        'miss_reasons': [None if hit else 'ranking' if present else 'split_across_chunks'
                                         for hit, present in zip(hits, available)],
                        'results': [{'document': Path(r.path).relative_to(root).as_posix(),
                                     'location': r.location, 'text': r.text, 'score': r.score} for r in rows]})
                variants.append({'chunking': asdict(config), 'pipeline_version': index.processing_version,
                    'chunks': index.status()['chunks'], 'index_ms': round(index_ms, 3),
                    'micro_literal_evidence_recall_at_5': sum(sum(c['hits']) for c in cases)/sum(len(c['hits']) for c in cases),
                    'all_evidence_question_rate': sum(c['all_evidence_found'] for c in cases)/len(cases),
                    'p50_search_ms': percentile(latencies, .5), 'p95_search_ms': percentile(latencies, .95),
                    'cases': cases})
            finally:
                index.close()
    return {'schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(), 'provenance': fixture['provenance'],
        'mode': 'offline', 'embedding': 'local:hash-256', 'retrieval': 'hybrid', 'rerank': False,
        'top_k': 5, 'lexical': 'enhanced-v1', 'prompt_version': None, 'generation': None,
        'api_usage': {'requests': 0, 'tokens': 0}, 'api_cost_usd': 0, 'local_compute_cost_usd': None,
        'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        'corpus_sha256': {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in fixture['documents'].items()},
        'corpus_characters': {name: len(text) for name, text in fixture['documents'].items()},
        'code_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((ROOT/'rag').glob('*.py')) + [Path(__file__).resolve()]},
        'limitations': ['AI-authored development cases; not a human-reviewed holdout or workplace outcome.',
                        'Complete literal span must occur in one retrieved chunk; jointly reconstructable split spans count as misses.',
                        'Recall is evidence retrieval, not semantic citation precision or answer correctness.',
                        'Small sample, fixed runs first, first query cold; latency is exploratory.',
                        'Markdown only; no claim about PDF tables or other parsers.'], 'variants': variants}


def main():
    parser = argparse.ArgumentParser(description='Offline controlled chunking comparison using raw evidence labels')
    parser.add_argument('--fixture', type=Path, default=ROOT/'data/eval/chunking_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/chunking_comparison.json')
    parser.add_argument('--chunk-size', type=int, default=900)
    parser.add_argument('--chunk-overlap', type=int, default=160)
    args = parser.parse_args()
    report = compare(args.fixture, args.chunk_size, args.chunk_overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for variant in report['variants']:
        print(variant['chunking']['strategy'], variant['micro_literal_evidence_recall_at_5'], variant['chunks'])
    print('CHUNKING COMPARISON RECORDED')


if __name__ == '__main__':
    main()
