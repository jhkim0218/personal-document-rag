import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app
from rag.answer import Answerer
from rag.service import RAGService


class RuntimeModeTests(unittest.TestCase):
    def test_offline_ignores_keys_through_background_index_search_and_answer(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'OPENAI_API_KEY': 'never-send-this'}), \
                patch.object(socket.socket, 'connect', side_effect=AssertionError('external network')) as connect:
            root = Path(temporary)
            (root/'guide.md').write_text('# Retrieval\nRRF combines keyword and vector rankings.', encoding='utf-8')
            service = RAGService(root, root/'index.db', mode='offline')
            try:
                self.assertEqual(service.index_documents().indexed, 1)
                self.assertEqual(service.index_documents().skipped, 1)
                self.assertTrue(service.search('RRF'))
                answer = service.ask('What does RRF combine?')
                self.assertIn('[1]', answer.text)
                self.assertEqual(answer.status, 'fallback')
                self.assertIsNone(answer.error)
                status = service.status()
                self.assertEqual(status['runtime_mode'], 'offline')
                self.assertEqual(status['embedding_mode'], 'local:hash-256')
                self.assertEqual(status['generation_mode'], 'local:extractive')
                self.assertFalse(status['external_transmission'])
                self.assertTrue(status['api_usage']['persistent'])
                connect.assert_not_called()
            finally:
                service.close()

    def test_auto_reports_external_mode_without_making_requests(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            service = RAGService(temporary, Path(temporary)/'index.db')
            try:
                self.assertTrue(service.status()['external_transmission'])
                self.assertTrue(service.status()['generation_mode'].startswith('openai:'))
            finally:
                service.close()

    def test_invalid_mode_or_custom_offline_answerer_rejected(self):
        for options in ({'mode': 'typo'}, {'mode': 'offline', 'answerer': Answerer(api_key='')}):
            with self.assertRaises(ValueError):
                RAGService('unused', 'unused.db', **options)

    def test_changed_embedding_mode_requires_reindex_before_hybrid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'guide.txt').write_text('RRF combines rankings.', encoding='utf-8')
            service = RAGService(root, root/'index.db', mode='offline')
            try:
                service.index_documents()
                service.index.connection.execute("UPDATE documents SET embedding_mode='openai:old-model'")
                service.index.connection.commit()
                with self.assertRaisesRegex(ValueError, 'Re-index'):
                    service.search('RRF')
                self.assertTrue(service.search('RRF', mode='keyword'))
                self.assertEqual(service.index_documents().indexed, 1)
                self.assertTrue(service.search('RRF'))
            finally:
                service.close()

    def test_cli_browser_flag_uses_bound_port_and_preserves_cleanup_on_launch_failure(self):
        server = Mock()
        server.server_address = ('127.0.0.1', 43210)
        server.service.status.return_value = {'runtime_mode': 'offline', 'embedding_mode': 'local:hash-256',
                                              'generation_mode': 'local:extractive', 'external_transmission': False}
        with patch('sys.argv', ['app.py', '--mode', 'offline', '--port', '0', '--open-browser']), \
                patch('app.create_server', return_value=server) as create, \
                patch('app.webbrowser.open', side_effect=OSError('no browser')) as browser:
            app.main()
        self.assertEqual(create.call_args.kwargs['mode'], 'offline')
        browser.assert_called_once_with('http://127.0.0.1:43210')
        server.serve_forever.assert_called_once()
        server.service.close.assert_called_once()
        server.server_close.assert_called_once()

    def test_cli_passes_explicit_local_model_directories(self):
        server = Mock()
        server.server_address = ('127.0.0.1', 43210)
        server.service.status.return_value = {'runtime_mode': 'local', 'embedding_mode': 'local:hash-256',
                                              'generation_mode': 'local:extractive', 'external_transmission': False}
        with patch('sys.argv', ['app.py', '--mode', 'local', '--local-reranker-model', 'D:/models/reranker']), patch('app.create_server', return_value=server) as create:
            app.main()
        self.assertEqual(create.call_args.kwargs['mode'], 'local')
        self.assertEqual(create.call_args.kwargs['local_reranker_model'], 'D:/models/reranker')
        server.service.close.assert_called_once()

    def test_cli_passes_explicit_pricing_file(self):
        server = Mock()
        server.server_address = ('127.0.0.1', 43210)
        server.service.status.return_value = {'runtime_mode': 'offline', 'embedding_mode': 'local:hash-256',
                                              'generation_mode': 'local:extractive', 'external_transmission': False}
        with patch('sys.argv', ['app.py', '--mode', 'offline', '--pricing', 'D:/private/pricing.json']), patch('app.create_server', return_value=server) as create:
            app.main()
        self.assertEqual(create.call_args.kwargs['pricing_path'], 'D:/private/pricing.json')
        server.service.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
