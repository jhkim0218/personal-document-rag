from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def analyze(report_path: Path, questions_path: Path, limit: int = 5) -> dict:
    report = json.loads(report_path.read_text(encoding='utf-8'))
    questions = {item['id']: item for item in (json.loads(line) for line in questions_path.read_text(encoding='utf-8').splitlines() if line.strip())}
    if not isinstance(report.get('variants'), list) or limit < 1:
        raise ValueError('Evaluation report variants and positive limit are required')
    failures = []
    for variant in report['variants']:
        for case in variant.get('cases', []):
            question = questions.get(case.get('id'))
            if not question:
                raise ValueError(f"Evaluation case has no question label: {case.get('id')}")
            if question['answerable'] and not case['hit']:
                kind = 'retrieval_or_abstention_failure'
            elif not question['answerable'] and case['answer_status'] != 'abstained':
                kind = 'unsupported_answer_not_abstained'
            else:
                continue
            failures.append({'id': question['id'], 'question': question['question'], 'answerable': question['answerable'],
                             'expected_document': question['expected_document'], 'answer_status': case['answer_status'],
                             'top_result': case['top_result'], 'configuration': {'mode': variant['mode'], 'rerank': variant['rerank']},
                             'observed_failure': kind, 'root_cause': None, 'change_applied': None,
                             'before_after_result': None, 'review_state': 'needs_human_investigation'})
    if len(failures) < limit:
        raise ValueError(f'Only {len(failures)} failure cases available; need {limit}')
    return {'schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'evaluation_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
            'questions_sha256': hashlib.sha256(questions_path.read_bytes()).hexdigest(),
            'selected_failure_count': limit, 'failures': failures[:limit],
            'limitations': ['Cases are selected from AI-authored development evaluation, not a human-reviewed holdout.',
                            'root_cause, change_applied, and before_after_result remain null until a human investigates each case.',
                            'This queue does not prove a quality improvement or workplace outcome.']}


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a review queue from reproducible RAG evaluation failures')
    parser.add_argument('--report', type=Path, default=ROOT/'results/evaluation.json')
    parser.add_argument('--questions', type=Path, default=ROOT/'data/eval/questions.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'results/failure_analysis.json')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    analysis = analyze(args.report, args.questions, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FAILURE ANALYSIS QUEUE PASSED:', len(analysis['failures']))


if __name__ == '__main__':
    main()
