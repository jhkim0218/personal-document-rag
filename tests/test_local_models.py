from __future__ import annotations

import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.embeddings import Embeddings
from rag.index import RAGIndex
from rag.local_models import LocalReranker
from rag.service import RAGService


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **options):
        self.calls.append((texts, options))
        return [[1.0, 0.0] if 'alpha' in text else [0.0, 1.0] for text in texts]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, **options):
        self.calls.append((pairs, options))
        return [10.0 if 'strong' in text else -10.0 for _, text in pairs]


class LocalModelTests(unittest.TestCase):
    def test_adapters_explicitly_request_local_files_only(self):
        sentence_calls, reranker_calls = [], []

        class SentenceTransformer:
            def __init__(self, path, **options):
                sentence_calls.append((path, options))

            def encode(self, texts, **options):
                return [[1.0, 0.0] for _ in texts]

        class CrossEncoder:
            def __init__(self, path, **options):
                reranker_calls.append((path, options))

            def predict(self, pairs, **options):
                return [1.0 for _ in pairs]

        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {'sentence_transformers': types.SimpleNamespace(SentenceTransformer=SentenceTransformer, CrossEncoder=CrossEncoder)}):
            self.assertEqual(Embeddings(api_key='', local_model_path=temporary).embed(['alpha']), [[1.0, 0.0]])
            self.assertEqual(LocalReranker(temporary).scores('alpha', ['alpha evidence']), [1.0])
        self.assertTrue(sentence_calls[0][1]['local_files_only'])
        self.assertTrue(reranker_calls[0][1]['local_files_only'])

    def test_local_encoder_uses_existing_directory_without_network(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(socket.socket, 'connect', side_effect=AssertionError('network')) as connect:
            encoder = FakeEncoder()
            embeddings = Embeddings(api_key='', local_model_path=temporary, local_encoder=encoder)
            self.assertEqual(embeddings.mode, f'local:sentence-transformers:{Path(temporary).name}')
            self.assertEqual(embeddings.embed(['alpha', 'beta']), [[1.0, 0.0], [0.0, 1.0]])
            self.assertTrue(encoder.calls[0][1]['normalize_embeddings'])
            connect.assert_not_called()

    def test_local_reranker_reorders_candidates_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'weak.md').write_text('alpha weak evidence', encoding='utf-8')
            (root/'strong.md').write_text('alpha strong evidence', encoding='utf-8')
            fake = FakeReranker()
            index = RAGIndex(root/'index.db', Embeddings(api_key=''), reranker=LocalReranker(root, fake))
            try:
                index.index_directory(root)
                ranked = index.search('alpha', limit=2, mode='keyword', rerank=True)
                self.assertEqual(Path(ranked[0].path).name, 'strong.md')
                self.assertEqual(len(fake.calls), 1)
                index.search('alpha', limit=2, mode='keyword', rerank=False)
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(index.status()['reranker_mode'], f'local:cross-encoder:{root.name}')
            finally:
                index.close()

    def test_local_mode_is_explicit_and_rejects_missing_or_ambiguous_models(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict('os.environ', {'OPENAI_API_KEY': 'never-send-this'}):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'requires a local'):
                RAGService(root, root/'none.db', mode='local')
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                Embeddings(api_key='', local_model_path=str(root/'missing'))
            with self.assertRaisesRegex(ValueError, 'either'):
                Embeddings(api_key='key', local_model_path=str(root))
            service = RAGService(root, root/'local.db', mode='local', local_reranker_model=str(root))
            try:
                self.assertEqual(service.status()['runtime_mode'], 'local')
                self.assertFalse(service.status()['external_transmission'])
                self.assertEqual(service.status()['embedding_mode'], 'local:hash-256')
            finally:
                service.close()
