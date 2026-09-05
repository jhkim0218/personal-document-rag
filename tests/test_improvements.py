from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.embeddings import Embeddings
from rag.index import RAGIndex
from rag.service import RAGService
from scripts.run_eval import main as evaluate_main


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / 'a', self.root / 'b'
        self.a.mkdir()
        self.b.mkdir()
        (self.a / 'alpha.txt').write_text('Alpha launch Friday.', encoding='utf-8')
        (self.b / 'beta.txt').write_text('Beta launch Monday.', encoding='utf-8')
        self.db = self.root / 'index.db'

    def tearDown(self):
        self.temp.cleanup()

    def test_active_folder_limits_search_counts_and_citations(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            a = RAGService(self.a, self.db)
            a.index_documents()
            old_id = a.search('launch')[0]['chunk_id']
            a.close()
            b = RAGService(self.b, self.db)
            try:
                b.index_documents()
                self.assertEqual(b.status()['documents'], 1)
                self.assertTrue(all(Path(r['path']).parent == self.b for r in b.search('launch')))
                self.assertIsNone(b.source(old_id))
            finally:
                b.close()

    def test_failed_embedding_preserves_existing_evidence_and_retry_updates(self):
        index = RAGIndex(self.db, embeddings=Embeddings(api_key=''))
        try:
            index.index_directory(self.a)
            old = index.search('launch')[0]
            (self.a / 'alpha.txt').write_text('Alpha launch Tuesday.', encoding='utf-8')
            with patch.object(index.embeddings, 'embed', side_effect=ValueError('Injected failure')):
                failed = index.index_directory(self.a)
            self.assertEqual(len(failed.failed), 1)
            self.assertEqual(index.source(old.chunk_id)['text'], 'Alpha launch Friday.')
            self.assertEqual(index.index_directory(self.a).indexed, 1)
            self.assertIsNone(index.source(old.chunk_id))
            self.assertIn('Tuesday', index.search('launch')[0].text)
        finally:
            index.close()

    def test_sql_write_failure_rolls_back_document_and_fts(self):
        index = RAGIndex(self.db, embeddings=Embeddings(api_key=''))
        try:
            index.index_directory(self.a)
            old = index.search('launch')[0]
            (self.a / 'alpha.txt').write_text('Alpha launch Tuesday.', encoding='utf-8')
            index.connection.execute("CREATE TRIGGER inject_failure BEFORE INSERT ON chunks BEGIN SELECT RAISE(ABORT, 'injected'); END")
            failed = index.index_directory(self.a)
            self.assertEqual(len(failed.failed), 1)
            self.assertEqual(index.source(old.chunk_id)['text'], old.text)
            self.assertEqual(index.status()['documents'], 1)
            if index.fts_enabled:
                self.assertEqual(index.connection.execute('SELECT COUNT(*) FROM chunks_fts').fetchone()[0], 1)
        finally:
            index.close()

    def test_unchanged_documents_never_reparse_or_embed_and_keyword_needs_no_api(self):
        index = RAGIndex(self.db, embeddings=Embeddings(api_key=''))
        try:
            index.index_directory(self.a)
            with patch('rag.index.parse_document', side_effect=AssertionError('Reparsed unchanged file')), patch.object(index.embeddings, 'embed', side_effect=AssertionError('Network requested')):
                self.assertEqual(index.index_directory(self.a).skipped, 1)
                self.assertTrue(index.search('launch', mode='keyword'))
        finally:
            index.close()

    def test_busy_service_rejects_duplicate_index(self):
        service = RAGService(self.a, self.db)
        held, release = threading.Event(), threading.Event()
        def hold():
            with service.lock:
                held.set()
                release.wait(5)
        thread = threading.Thread(target=hold)
        thread.start()
        try:
            self.assertTrue(held.wait(2))
            with self.assertRaises(RuntimeError):
                service.index_documents()
        finally:
            release.set()
            thread.join()
            service.close()

    def test_offline_evaluation_ignores_api_key_and_labels_unmeasured_grounding(self):
        output = self.root / 'evaluation.json'
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'must-not-use'}), patch('urllib.request.urlopen', side_effect=AssertionError('Offline network call')), patch('sys.argv', ['run_eval', '--data', 'data/sample', '--questions', 'data/eval/questions.jsonl', '--output', str(output)]):
            evaluate_main()
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(report['run']['api_calls'], 0)
        self.assertEqual(len(report['variants']), 4)
        for variant in report['variants']:
            self.assertIsNone(variant['grounded_answer_rate'])
            self.assertIsNone(variant['semantic_citation_precision'])
            self.assertIn('document_hit_at_5', variant)
            self.assertNotIn('citation_precision', variant)


if __name__ == '__main__':
    unittest.main()
