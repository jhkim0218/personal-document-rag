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

from rag.embeddings import Embeddings
from rag.index import RAGIndex
from rag.relations import Relations


def relative(path: str, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def evaluate(fixture_path: Path) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    if not isinstance(fixture.get('documents'), dict) or not isinstance(fixture.get('queries'), list) or not fixture['queries']:
        raise ValueError('Fixture needs documents and at least one query')
    with tempfile.TemporaryDirectory() as temporary, patch.object(socket.socket, 'connect', side_effect=AssertionError('Relation evaluation attempted network')):
        root = Path(temporary)/'corpus'
        root.mkdir()
        for name, text in fixture['documents'].items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()) or target.suffix != '.md' or not isinstance(text, str):
                raise ValueError('Fixture documents must be Markdown files inside corpus')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding='utf-8')
        index = RAGIndex(Path(temporary)/'relations.sqlite3', Embeddings(api_key=''))
        try:
            summary = index.index_directory(root)
            if summary.failed or summary.indexed != len(fixture['documents']):
                raise AssertionError(f'Fixture indexing failed: {summary.failed}')
            configured = json.loads(json.dumps(fixture['relations']))
            for project in configured['projects']:
                for decision in project['decisions']:
                    decision['documents'] = [str((root/path).resolve()) for path in decision['documents']]
            relations = Relations(Path(temporary)/'relations.json')
            relations.configure(configured, index.document_source)
            cases = []
            for query in fixture['queries']:
                gold = query.get('gold_documents')
                if not isinstance(query.get('id'), str) or not isinstance(query.get('search_query'), str) or not isinstance(query.get('relation_query'), str) or not isinstance(gold, list) or not gold or any(path not in fixture['documents'] for path in gold):
                    raise ValueError('Invalid relation query labels')
                retrieved = [relative(result.path, root) for result in index.search(query['search_query'], limit=5, mode='hybrid', rerank=True)]
                view = relations.view(query['relation_query'], index.document_source)
                mapped = [relative(document['path'], root) for project in view['projects'] for decision in project['decisions'] for document in decision['documents']]
                search_hits = [path in retrieved for path in gold]
                relation_hits = [path in mapped for path in gold]
                cases.append({'id': query['id'], 'gold_documents': gold, 'search_documents': retrieved,
                              'relation_documents': mapped, 'search_coverage': sum(search_hits)/len(gold),
                              'relation_coverage': sum(relation_hits)/len(gold), 'relation_unavailable_documents': view['unavailable_documents']})
        finally:
            index.close()
    return {'schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'offline',
            'embedding': 'local:hash-256', 'top_k': 5, 'api_usage': {'requests': 0, 'tokens': 0},
            'api_cost_usd': 0, 'local_compute_cost_usd': None,
            'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(), 'provenance': fixture['provenance'],
            'metrics': {'search_document_coverage': sum(case['search_coverage'] for case in cases)/len(cases),
                        'explicit_relation_document_coverage': sum(case['relation_coverage'] for case in cases)/len(cases)},
            'cases': cases,
            'limitations': ['Relations are explicit fixture configuration, not extracted entities or semantic links.',
                            'This is AI-authored development data, not a human-reviewed holdout or workplace result.',
                            'Coverage only tests the listed documents, not whether they contain sufficient evidence.']}


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare explicit relation-map document coverage with top-k search offline')
    parser.add_argument('--fixture', type=Path, default=ROOT/'data/eval/relations_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/relations_evaluation.json')
    args = parser.parse_args()
    report = evaluate(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RELATION EVALUATION PASSED:', report['metrics'])


if __name__ == '__main__':
    main()
