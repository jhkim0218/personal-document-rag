from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock


MAX_RESPONSE_BYTES = 2_000_000


def load_pricing(path: str | Path | None) -> dict | None:
    """Load explicit, dated per-million-token rates without network access."""
    if path is None:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Invalid pricing file: {error}') from error
    if not isinstance(raw, dict) or not isinstance(raw.get('effective_date'), str) or not isinstance(raw.get('models'), dict):
        raise ValueError('Pricing file requires effective_date and models')
    try:
        datetime.strptime(raw['effective_date'], '%Y-%m-%d')
    except ValueError as error:
        raise ValueError('Pricing effective_date must use YYYY-MM-DD') from error
    models = {}
    for model, rates in raw['models'].items():
        if not isinstance(model, str) or not model or not isinstance(rates, dict):
            raise ValueError('Pricing models must map nonempty model names to rate objects')
        cleaned = {}
        for key in ('input_usd_per_million_tokens', 'output_usd_per_million_tokens'):
            value = rates.get(key)
            if value is not None:
                if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid {key} for {model}')
                cleaned[key] = float(value)
        if not cleaned:
            raise ValueError(f'Pricing for {model} needs an input or output rate')
        models[model] = cleaned
    return {'effective_date': raw['effective_date'], 'models': models}


def estimate_cost(model: str, usage: dict | None, pricing: dict | None) -> tuple[float | None, dict | None]:
    if not usage or not pricing or model not in pricing['models']:
        return None, None
    rates = pricing['models'][model]
    input_tokens = usage.get('input_tokens', usage.get('prompt_tokens'))
    output_tokens = usage.get('output_tokens', usage.get('completion_tokens'))
    if type(input_tokens) is not int or input_tokens < 0:
        return None, None
    if 'output_usd_per_million_tokens' in rates and (type(output_tokens) is not int or output_tokens < 0):
        return None, None
    cost = input_tokens * rates.get('input_usd_per_million_tokens', 0) / 1_000_000
    if output_tokens is not None:
        cost += output_tokens * rates.get('output_usd_per_million_tokens', 0) / 1_000_000
    basis = {'effective_date': pricing['effective_date'], 'per': 'million_tokens', **rates}
    return round(cost, 12), basis


def retry_delay(header, attempt):
    try:
        delay = float(header)
    except (ValueError, TypeError):
        try:
            delay = parsedate_to_datetime(header).timestamp() - time.time()
        except (ValueError, TypeError, AttributeError, OverflowError):
            delay = -1
    if not math.isfinite(delay) or delay < 0:
        delay = 2 ** attempt
    return delay + random.uniform(0, .25)


class APIRequests:
    """Bounded retries shared by embeddings and answers, with content-free attempt records."""

    def __init__(self, journal=None, pricing=None):
        self.records = deque(maxlen=100)
        self.lock = Lock()
        self.journal = journal
        self.pricing = pricing

    def snapshot(self):
        with self.lock:
            return [dict(record) for record in self.records]

    @staticmethod
    def _read_response(response) -> dict:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError(f'API response exceeds {MAX_RESPONSE_BYTES} byte limit')
        decoded = json.loads(body.decode('utf-8'))
        if not isinstance(decoded, dict):
            raise ValueError('API response must be an object')
        return decoded

    def send(self, request):
        model = json.loads(request.data)['model']
        started = time.monotonic()
        for attempt in range(3):
            remaining = 45 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('API retry time budget exhausted')
            record = {'time_utc': datetime.now(timezone.utc).isoformat(), 'model': model,
                      'endpoint': request.full_url.rsplit('/', 1)[-1], 'attempt': attempt + 1,
                      'status': None, 'usage': None, 'estimated_cost_usd': None, 'pricing': None}
            delay = None
            attempt_started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=remaining) as response:
                    body = self._read_response(response)
                record['status'] = 'received'
                usage = body.get('usage')
                if isinstance(usage, dict):
                    record['usage'] = {key: value for key, value in usage.items()
                                       if key in {'input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens'}
                                       and type(value) is int and value >= 0} or None
                record['estimated_cost_usd'], record['pricing'] = estimate_cost(model, record['usage'], self.pricing)
                return body
            except urllib.error.HTTPError as error:
                record['status'] = error.code
                retryable = error.code in {500, 502, 503, 504}
                if error.code == 429:
                    try:
                        detail = json.loads(error.read()).get('error', {})
                        retryable = detail.get('code') == 'rate_limit_exceeded' or detail.get('type') == 'rate_limit_error'
                        if detail.get('code') in {'insufficient_quota', 'billing_hard_limit_reached'}:
                            retryable = False
                    except (ValueError, AttributeError):
                        retryable = False  # Unknown 429 may need account action; do not guess.
                delay = retry_delay(error.headers.get('Retry-After') if error.headers else None, attempt)
                error.close()
                if not retryable or attempt == 2 or delay >= 45 - (time.monotonic() - started):
                    raise
            except TimeoutError:
                record['status'] = 'timeout'
                delay = retry_delay(None, attempt)
                if attempt == 2 or delay >= 45 - (time.monotonic() - started):
                    raise
            except Exception:
                record['status'] = 'failed'
                raise
            finally:
                record['elapsed_ms'] = round((time.monotonic() - attempt_started) * 1000, 3)
                with self.lock:
                    self.records.append(record)
                if self.journal:
                    self.journal.record(record)
            time.sleep(delay)
