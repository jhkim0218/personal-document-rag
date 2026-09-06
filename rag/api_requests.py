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
from threading import Lock


MAX_RESPONSE_BYTES = 2_000_000


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

    def __init__(self, journal=None):
        self.records = deque(maxlen=100)
        self.lock = Lock()
        self.journal = journal

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
                      'status': None, 'usage': None, 'estimated_cost_usd': None}
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
