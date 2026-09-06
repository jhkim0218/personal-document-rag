import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from rag.api_requests import APIRequests, MAX_RESPONSE_BYTES, load_pricing, retry_delay
from rag.embeddings import Embeddings
from rag.answer import Answerer
from tests.test_answer import result


def response(body):
    return io.BytesIO(json.dumps(body).encode())


def failure(code, error_code='', retry_after='0'):
    return urllib.error.HTTPError('https://api.openai.com/v1/embeddings', code, 'API error',
        {'Retry-After': retry_after}, response({'error': {'code': error_code}}))


class APIRequestTests(unittest.TestCase):
    def request(self):
        return urllib.request.Request('https://api.openai.com/v1/embeddings',
            data=json.dumps({'model': 'test', 'input': 'private-document'}).encode())

    def test_temporary_rate_limit_retries_and_records_only_usage(self):
        client = APIRequests()
        body = {'data': [], 'usage': {'prompt_tokens': 12, 'total_tokens': 12, 'private': 'secret'}}
        with patch('rag.api_requests.urllib.request.urlopen', side_effect=[failure(429, 'rate_limit_exceeded', '2'), response(body)]) as opened, \
                patch('rag.api_requests.time.sleep') as sleep, patch('rag.api_requests.random.uniform', return_value=.1):
            self.assertEqual(client.send(self.request()), body)
        self.assertEqual(opened.call_count, 2)
        sleep.assert_called_once_with(2.1)
        records = client.snapshot()
        self.assertIsNone(records[0]['usage'])
        self.assertEqual(records[1]['usage'], {'prompt_tokens': 12, 'total_tokens': 12})
        self.assertNotIn('private-document', json.dumps(records))
        self.assertNotIn('secret', json.dumps(records))
        self.assertIsNone(records[1]['estimated_cost_usd'])

    def test_permanent_quota_unknown_rate_limit_and_long_wait_do_not_retry(self):
        for error in (failure(401), failure(400), failure(429, 'insufficient_quota'), failure(429), failure(503, retry_after='120')):
            with patch('rag.api_requests.urllib.request.urlopen', side_effect=error) as opened, patch('rag.api_requests.time.sleep') as sleep:
                with self.assertRaises(urllib.error.HTTPError):
                    APIRequests().send(self.request())
            self.assertEqual(opened.call_count, 1)
            sleep.assert_not_called()

    def test_repeated_server_failures_and_timeouts_stop_at_three_attempts(self):
        for errors in ([failure(503) for _ in range(3)], [TimeoutError('timeout') for _ in range(3)]):
            client = APIRequests()
            with patch('rag.api_requests.urllib.request.urlopen', side_effect=errors) as opened, patch('rag.api_requests.time.sleep'):
                with self.assertRaises((urllib.error.HTTPError, TimeoutError)):
                    client.send(self.request())
            self.assertEqual(opened.call_count, 3)
            self.assertEqual(len(client.snapshot()), 3)

    def test_retry_after_date_invalid_header_and_time_budget(self):
        with patch('rag.api_requests.random.uniform', return_value=0), patch('rag.api_requests.time.time', return_value=0):
            self.assertEqual(retry_delay('Thu, 01 Jan 1970 00:00:05 GMT', 0), 5)
            self.assertEqual(retry_delay('nan', 1), 2)
            self.assertEqual(retry_delay('garbage', 1), 2)
        with patch('rag.api_requests.time.monotonic', side_effect=[0, 46]), patch('rag.api_requests.urllib.request.urlopen') as opened:
            with self.assertRaises(TimeoutError):
                APIRequests().send(self.request())
        opened.assert_not_called()

    def test_embedding_and_answer_paths_share_retry_and_usage_behavior(self):
        embedding = Embeddings(api_key='test')
        with patch('rag.api_requests.urllib.request.urlopen', side_effect=[failure(503), response({'data': [{'index': 0, 'embedding': [1.0]}], 'usage': {'total_tokens': 4}})]), patch('rag.api_requests.time.sleep'):
            self.assertEqual(embedding.embed(['text']), [[1.0]])
        self.assertEqual(embedding.requests.snapshot()[-1]['usage']['total_tokens'], 4)
        answerer = Answerer(api_key='test')
        with patch('rag.api_requests.urllib.request.urlopen', side_effect=[failure(503), response({'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Launch Friday. [1]'}]}], 'usage': {'input_tokens': 5, 'output_tokens': 6}})]), patch('rag.api_requests.time.sleep'):
            answer = answerer.answer('When is launch?', [result('Launch Friday.')])
        self.assertEqual(answer.status, 'answered')
        self.assertEqual(answerer.requests.snapshot()[-1]['usage']['output_tokens'], 6)

    def test_response_read_is_bounded_and_oversized_payload_is_rejected(self):
        class OversizedResponse:
            def __init__(self): self.read_size = None
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1):
                self.read_size = size
                return b'x' * (MAX_RESPONSE_BYTES + 1)

        response = OversizedResponse()
        client = APIRequests()
        with patch('rag.api_requests.urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                client.send(self.request())
        self.assertEqual(response.read_size, MAX_RESPONSE_BYTES + 1)
        self.assertEqual(client.snapshot()[-1]['status'], 'failed')

    def test_explicit_dated_rates_calculate_only_observed_token_types(self):
        pricing = {'effective_date': '2026-09-06', 'models': {'test': {
            'input_usd_per_million_tokens': 2, 'output_usd_per_million_tokens': 5}}}
        client = APIRequests(pricing=pricing)
        body = {'data': [], 'usage': {'input_tokens': 10, 'output_tokens': 2}}
        with patch('rag.api_requests.urllib.request.urlopen', return_value=response(body)):
            client.send(self.request())
        record = client.snapshot()[0]
        self.assertEqual(record['estimated_cost_usd'], 0.00003)
        self.assertEqual(record['pricing'], {'effective_date': '2026-09-06', 'per': 'million_tokens',
                                             'input_usd_per_million_tokens': 2.0, 'output_usd_per_million_tokens': 5.0})
        unknown = APIRequests(pricing={'effective_date': '2026-09-06', 'models': {}})
        with patch('rag.api_requests.urllib.request.urlopen', return_value=response(body)):
            unknown.send(self.request())
        self.assertIsNone(unknown.snapshot()[0]['estimated_cost_usd'])

    def test_pricing_file_requires_dated_model_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'pricing.json'
            path.write_text(json.dumps({'effective_date': '2026-09-06', 'models': {'test': {'input_usd_per_million_tokens': 1}}}), encoding='utf-8')
            self.assertEqual(load_pricing(path)['models']['test']['input_usd_per_million_tokens'], 1.0)
            path.write_text('{"models":{}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'effective_date'):
                load_pricing(path)
            path.write_text('{"effective_date":"today","models":{}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'YYYY-MM-DD'):
                load_pricing(path)
