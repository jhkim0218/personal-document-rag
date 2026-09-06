from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from rag.api_requests import APIRequests
from rag.usage import APIUsageJournal


class UsageTests(unittest.TestCase):
    def test_content_free_attempt_persists_across_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'usage.sqlite3'
            journal = APIUsageJournal(path)

            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self, size=-1): return b'{"usage":{"input_tokens":3,"output_tokens":2}}'

            request = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps({'model':'test-model','secret':'never-store'}).encode(), method='POST')
            with patch('rag.api_requests.urllib.request.urlopen', return_value=Response()):
                APIRequests(journal).send(request)
            record = journal.status()['records'][0]
            self.assertEqual(record['model'], 'test-model')
            self.assertEqual(record['usage'], {'input_tokens': 3, 'output_tokens': 2})
            self.assertNotIn('secret', str(record))
            self.assertNotIn('never-store', path.read_text(encoding='utf-8', errors='ignore'))
            journal.close()
            reopened = APIUsageJournal(path)
            self.assertEqual(len(reopened.status()['records']), 1)
            reopened.close()
