from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.answer import Answerer
from rag.embeddings import Embeddings
from rag.index import RAGIndex
from rag.text import split_sentences
from scripts.compare_chunking import validate_labels


def relative(path: str, root: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def source_matches_gold(source: dict, gold: dict, root: Path) -> bool:
    return relative(str(source['path']), root) == gold['document'] and gold['text'] in str(source.get('excerpt', ''))


def literal_cited_claim_support(answer_text: str, sources: list[dict]) -> tuple[int, int]:
    """Exact-sentence check for labeled offline fallback; not paraphrase entailment."""
    by_number = {int(source['number']): str(source['context']) for source in sources}
    supported = total = 0
    attached = re.sub(r"([.!?。])\s+((?:\[\d+\]\s*)+)", r"\1\2", answer_text)
    for sentence in split_sentences(attached):
        numbers = [int(value) for value in re.findall(r'\[(\d+)\]', sentence)]
        if not numbers:
            continue
        total += 1
        claim = re.sub(r'\s*\[\d+\]', '', sentence).strip(' -\t')
        if any(claim and claim in by_number.get(number, '') for number in numbers):
            supported += 1
    return supported, total


def evaluate(fixture_path: Path) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as temporary, patch.object(socket.socket, 'connect', side_effect=AssertionError('Grounding evaluation attempted network')):
        root = Path(temporary)/'corpus'
        root.mkdir()
        for name, text in fixture['documents'].items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()) or target.suffix != '.md':
                raise ValueError('Fixture documents must be Markdown paths inside corpus')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding='utf-8')
        validate_labels(fixture, root)
        index = RAGIndex(Path(temporary)/'grounding.sqlite3', Embeddings(api_key=''))
        try:
            summary = index.index_directory(root)
            if summary.failed:
                raise AssertionError(f'Fixture indexing failed: {summary.failed}')
            answerer = Answerer(api_key='')
            cases, retrieved, cited = [], [], []
            literal_supported = literal_total = 0
            for case in fixture['queries']:
                results = index.search(case['question'], limit=5, mode='hybrid', rerank=True)
                answer = answerer.answer(case['question'], results)
                retrieved_hits = [any(relative(row.path, root) == gold['document'] and gold['text'] in row.text for row in results) for gold in case['gold']]
                cited_hits = [any(source_matches_gold(source, gold, root) for source in answer.sources) for gold in case['gold']]
                supported, total = literal_cited_claim_support(answer.text, answer.sources)
                retrieved.extend(retrieved_hits)
                cited.extend(cited_hits)
                literal_supported += supported
                literal_total += total
                cases.append({'id': case['id'], 'type': case['type'], 'gold_count': len(case['gold']),
                              'retrieved_gold': retrieved_hits, 'cited_gold': cited_hits,
                              'literal_cited_claims_supported': supported, 'literal_cited_claims_total': total,
                              'answer_status': answer.status, 'sources': [{'path': relative(str(s['path']), root), 'location': s['location'], 'excerpt': s['excerpt']} for s in answer.sources]})
        finally:
            index.close()
    return {'schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'offline',
            'embedding': 'local:hash-256', 'generation': 'local:extractive', 'top_k': 5, 'rerank': True,
            'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(), 'api_usage': {'requests': 0, 'tokens': 0},
            'api_cost_usd': 0, 'local_compute_cost_usd': None, 'provenance': fixture['provenance'],
            'metrics': {'literal_gold_evidence_recall_at_5': sum(retrieved)/len(retrieved),
                        'literal_gold_citation_coverage': sum(cited)/len(cited),
                        'literal_cited_claim_support_precision': literal_supported/literal_total if literal_total else None,
                        'semantic_citation_precision': None, 'grounded_answer_rate': None}, 'cases': cases,
            'limitations': ['Gold labels are AI-authored development labels in the input fixture, not human-reviewed semantic judgments.',
                            'A citation counts only when its displayed excerpt contains the labeled source span; a same-file wrong sentence fails.',
                            'Claim support uses exact sentence containment and rejects unsupported cited claims, but does not score paraphrase entailment.',
                            'semantic_citation_precision and grounded_answer_rate remain null until human-reviewed claim/evidence judgments exist.']}


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate literal evidence and cited-claim support without API calls')
    parser.add_argument('--fixture', type=Path, default=ROOT/'data/eval/chunking_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/grounding_evaluation.json')
    args = parser.parse_args()
    report = evaluate(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('GROUNDING EVALUATION PASSED:', report['metrics'])


if __name__ == '__main__':
    main()
