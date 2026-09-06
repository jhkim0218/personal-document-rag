import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.embeddings import Embeddings
from rag.index import RAGIndex


def batch_response(request):
    texts = json.loads(request.data)['input']
    return {'data': [{'index': i, 'embedding': [float(len(text)), 1.0]} for i, text in reversed(list(enumerate(texts)))]}


class EmbeddingBatchTests(unittest.TestCase):
    def test_count_and_byte_limits_preserve_order(self):
        for texts, counts in ((['x'*i for i in range(1, 131)], [64, 64, 2]), (['가'*2000]*11, [10, 1])):
            embedding = Embeddings(api_key='test')
            with patch.object(embedding.requests, 'send', side_effect=batch_response) as sent:
                vectors = embedding.embed(texts)
            self.assertEqual(vectors, [[float(len(text)), 1.0] for text in texts])
            batches = [json.loads(call.args[0].data)['input'] for call in sent.call_args_list]
            self.assertEqual([len(batch) for batch in batches], counts)
            self.assertEqual([text for batch in batches for text in batch], texts)
            self.assertTrue(all(sum(len(text.encode('utf-8')) for text in batch) <= 64000 for batch in batches))

    def test_all_inputs_validated_before_first_paid_request(self):
        embedding = Embeddings(api_key='test')
        for invalid in ('x'*8001, '가'*2667, '', ' ', None):
            with patch.object(embedding.requests, 'send') as sent:
                with self.assertRaises(ValueError):
                    embedding.embed(['valid']*65 + [invalid])
                sent.assert_not_called()

    def test_corrupt_response_rejected(self):
        embedding = Embeddings(api_key='test')
        for data in ([{'index': 0, 'embedding': [1]}, {'index': 0, 'embedding': [1]}],
                     [{'index': 0, 'embedding': [1]}, {'index': 2, 'embedding': [1]}],
                     [{'index': 0, 'embedding': [1]}, {'index': 1, 'embedding': [1, 2]}],
                     [{'index': 0, 'embedding': [True]}, {'index': 1, 'embedding': [1]}],
                     [{'index': 0, 'embedding': [float('nan')]}, {'index': 1, 'embedding': [1]}],
                     [{'index': 0, 'embedding': []}, {'index': 1, 'embedding': [1]}]):
            with patch.object(embedding.requests, 'send', return_value={'data': data}), self.assertRaises(ValueError):
                embedding.embed(['one', 'two'])
        calls = []
        def changed_dimensions(request):
            calls.append(request)
            body = batch_response(request)
            if len(calls) > 1:
                for item in body['data']:
                    item['embedding'].append(1.0)
            return body
        with patch.object(embedding.requests, 'send', side_effect=changed_dimensions), self.assertRaisesRegex(ValueError, 'between batches'):
            embedding.embed(['text']*65)

    def test_later_batch_failure_preserves_previous_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root/'guide.md'
            document.write_text('# Original\nOriginal launch Friday.', encoding='utf-8')
            embedding = Embeddings(api_key='test')
            index = RAGIndex(root/'index.db', embeddings=embedding)
            try:
                with patch.object(embedding.requests, 'send', side_effect=batch_response):
                    self.assertEqual(index.index_directory(root).indexed, 1)
                old = index.search('launch', mode='keyword')[0]
                document.write_text('\n'.join(f'# Section {i}\nRevised launch item {i}.' for i in range(70)), encoding='utf-8')
                calls = []
                def second_fails(request):
                    calls.append(request)
                    if len(calls) == 2:
                        raise TimeoutError('second batch timed out')
                    return batch_response(request)
                with patch.object(embedding.requests, 'send', side_effect=second_fails):
                    failed = index.index_directory(root)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(failed.failed), 1)
                self.assertEqual(index.source(old.chunk_id)['text'], old.text)
                with patch.object(embedding.requests, 'send', side_effect=batch_response):
                    self.assertEqual(index.index_directory(root).indexed, 1)
                self.assertIsNone(index.source(old.chunk_id))
            finally:
                index.close()
