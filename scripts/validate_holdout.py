from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read JSONL: {error}') from error


def normalized_question(value: object) -> str:
    return re.sub(r'\s+', ' ', str(value).strip()).casefold()


def validate(development: list[dict], holdout: list[dict]) -> dict:
    errors: list[str] = []
    dev_questions = {normalized_question(item.get('question')) for item in development}
    seen_ids: set[str] = set()
    reviewed = drafts = answerable = unanswerable = 0
    for number, item in enumerate(holdout, start=1):
        label = f'line {number}'
        identifier, question = item.get('id'), normalized_question(item.get('question'))
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            errors.append(f'{label}: id must be unique and nonempty')
        seen_ids.add(identifier) if isinstance(identifier, str) else None
        if not question:
            errors.append(f'{label}: question is required')
        elif question in dev_questions:
            errors.append(f'{label}: question overlaps the development set')
        if type(item.get('answerable')) is not bool:
            errors.append(f'{label}: answerable must be boolean')
            continue
        if item['answerable']:
            answerable += 1
            if not isinstance(item.get('expected_document'), str) or not item['expected_document']:
                errors.append(f'{label}: answerable case needs expected_document')
            if not isinstance(item.get('gold_location'), str) or not item['gold_location']:
                errors.append(f'{label}: answerable case needs gold_location')
            if not isinstance(item.get('required_facts'), list) or not item['required_facts'] or not all(isinstance(fact, str) and fact.strip() for fact in item['required_facts']):
                errors.append(f'{label}: answerable case needs nonempty required_facts')
        else:
            unanswerable += 1
        status = item.get('label_status')
        if status == 'human-reviewed':
            reviewed += 1
        elif status == 'ai-draft':
            drafts += 1
        else:
            errors.append(f'{label}: label_status must be human-reviewed or ai-draft')
    return {'valid': not errors, 'cases': len(holdout), 'answerable': answerable, 'unanswerable': unanswerable,
            'human_reviewed': reviewed, 'ai_draft': drafts, 'errors': errors}


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate a private RAG holdout without sending its content anywhere')
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--holdout', type=Path, required=True)
    parser.add_argument('--require-human-review', action='store_true')
    args = parser.parse_args()
    report = validate(load(args.development), load(args.holdout))
    if args.require_human_review and report['human_reviewed'] != report['cases']:
        report['errors'].append('Every holdout case must be human-reviewed for this run')
        report['valid'] = False
    print(json.dumps(report, ensure_ascii=False))
    if not report['valid']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
