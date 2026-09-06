from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def existing_executable(value: str) -> str | None:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.is_file() else shutil.which(value)


def check_runtime(requirements: set[str], *, embedding_model: str | None = None, reranker_model: str | None = None,
                  ocr_executable: str = 'tesseract', hwp_executable: str = 'hwp5txt') -> dict[str, object]:
    local_models = [path for path in (embedding_model, reranker_model) if path]
    checks = {
        'python': {'ready': sys.version_info >= (3, 11), 'detail': '.'.join(map(str, sys.version_info[:3]))},
        'local-model': {'ready': bool(local_models) and all(Path(path).is_dir() for path in local_models)
                        and importlib.util.find_spec('sentence_transformers') is not None,
                        'detail': 'requires an existing model directory and sentence-transformers'},
        'ocr': {'ready': bool(existing_executable(ocr_executable)), 'detail': f'executable: {ocr_executable}'},
        'hwp': {'ready': bool(existing_executable(hwp_executable)), 'detail': f'executable: {hwp_executable}'},
    }
    selected = ['python', *sorted(requirements)]
    return {'checks': {name: checks[name] for name in selected}, 'ready': all(checks[name]['ready'] for name in selected)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only readiness check for optional personal-document RAG features')
    parser.add_argument('--require', choices=('local-model', 'ocr', 'hwp'), action='append', default=[], help='Capability that must be ready')
    parser.add_argument('--local-embedding-model')
    parser.add_argument('--local-reranker-model')
    parser.add_argument('--ocr-executable', default='tesseract')
    parser.add_argument('--hwp-executable', default='hwp5txt')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = check_runtime(set(args.require), embedding_model=args.local_embedding_model, reranker_model=args.local_reranker_model,
                           ocr_executable=args.ocr_executable, hwp_executable=args.hwp_executable)
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        for name, result in report['checks'].items():
            print(f"{'READY' if result['ready'] else 'MISSING'} {name}: {result['detail']}")
        print('PREFLIGHT READY' if report['ready'] else 'PREFLIGHT INCOMPLETE')
    if not report['ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
