from __future__ import annotations

import argparse
import hashlib
import json
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


def evaluate(fixture_path: Path) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    if not isinstance(fixture.get('documents'), dict) or not isinstance(fixture.get('queries'), list) or not fixture['queries']:
        raise ValueError('Fixture needs documents and queries')
    with tempfile.TemporaryDirectory() as temporary, patch.object(socket.socket, 'connect', side_effect=AssertionError('Answer-policy evaluation attempted network')):
        root = Path(temporary)/'corpus'
        root.mkdir()
        for name, text in fixture['documents'].items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()) or target.suffix != '.md' or not isinstance(text, str):
                raise ValueError('Fixture documents must be Markdown inside corpus')
            target.write_text(text, encoding='utf-8')
        index = RAGIndex(Path(temporary)/'policy.sqlite3', Embeddings(api_key=''))
        try:
            summary = index.index_directory(root)
            if summary.failed:
                raise AssertionError(f'Fixture indexing failed: {summary.failed}')
            policies = {'permissive_literal': Answerer(api_key='', enforce_required_evidence=False),
                        'required_literal_partial': Answerer(api_key='', enforce_required_evidence=True)}
            cases = []
            for query in fixture['queries']:
                documents = query.get('documents')
                if not isinstance(query.get('id'), str) or not isinstance(query.get('question'), str) or type(query.get('requires_partial')) is not bool or not isinstance(documents, list) or not documents or any(path not in fixture['documents'] for path in documents):
                    raise ValueError('Invalid answer-policy query')
                results = [result for result in index.search(query['question'], limit=len(fixture['documents']), mode='hybrid', rerank=True) if Path(result.path).name in documents]
                statuses = {name: answerer.answer(query['question'], results).status for name, answerer in policies.items()}
                cases.append({'id': query['id'], 'documents': documents, 'requires_partial': query['requires_partial'], 'statuses': statuses})
        finally:
            index.close()
    required = [case for case in cases if case['requires_partial']]
    return {'schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'offline',
            'embedding': 'local:hash-256', 'generation': 'local:extractive', 'api_usage': {'requests': 0, 'tokens': 0},
            'api_cost_usd': 0, 'local_compute_cost_usd': None, 'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            'provenance': fixture['provenance'],
            'metrics': {'required_cases': len(required), 'required_literal_partial_rate': sum(case['statuses']['required_literal_partial'] == 'partial' for case in required)/len(required),
                        'permissive_literal_partial_rate': sum(case['statuses']['permissive_literal'] == 'partial' for case in required)/len(required)},
            'cases': cases,
            'limitations': ['Only explicit before/after, reason, p50, and p95 literal requirements are covered.',
                            'Each policy case uses an explicitly scoped fixture document set, so this isolates answer policy rather than measuring retrieval quality.',
                            'This does not measure semantic entailment, human claim review, or generated-answer correctness.',
                            'Fixture labels are AI-authored development data, not a holdout or workplace outcome.']}


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare literal partial-evidence answer policies offline')
    parser.add_argument('--fixture', type=Path, default=ROOT/'data/eval/answer_policy_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/answer_policy_evaluation.json')
    args = parser.parse_args()
    report = evaluate(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ANSWER POLICY EVALUATION PASSED:', report['metrics'])


if __name__ == '__main__':
    main()
