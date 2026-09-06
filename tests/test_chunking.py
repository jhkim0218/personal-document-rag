from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.documents import Chunking, ParsedDocument, ParsedSection, to_chunks
from rag.embeddings import Embeddings
from rag.index import RAGIndex
from rag.service import RAGService
from rag.text import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_fixed_crosses_sections_without_losing_text_or_locations(self):
        document = ParsedDocument(Path('example.md'), 'example', 'hash', [
            ParsedSection('first', 'abcdefghij'), ParsedSection('second', 'klmnopqrst')])
        chunks = to_chunks(document, 16, 4, 'fixed')
        self.assertEqual([text for _, text in chunks], ['abcdefghij\n\nklmn', 'klmnopqrst'])
        self.assertIn('first → second', chunks[0][0])
        self.assertIn('chars 12:22', chunks[1][0])
        self.assertEqual(chunks[0][1][-4:], chunks[1][1][:4])
        structured = to_chunks(document, 16, 4, 'structured')
        self.assertEqual(structured, [('first', 'abcdefghij'), ('second', 'klmnopqrst')])

    def test_long_section_bounds_and_content_coverage(self):
        text = ''.join(chr(0xAC00+i) for i in range(120))
        document = ParsedDocument(Path('long.md'), 'long', 'hash', [ParsedSection('section', text)])
        for strategy in ('fixed', 'structured'):
            chunks = to_chunks(document, 30, 5, strategy)
            self.assertTrue(all(0 < len(piece) <= 30 for _, piece in chunks))
            self.assertTrue(all('section' in location for location, _ in chunks))
            self.assertEqual(set(''.join(piece for _, piece in chunks)), set(text))

    def test_invalid_configuration_fails_before_chunking(self):
        for size, overlap in ((0, 0), (10, -1), (10, 10), (True, 0), (10, False)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    Chunking(max_chars=size, overlap_chars=overlap)
                with self.assertRaises(ValueError):
                    chunk_text('text', size, overlap)
        with self.assertRaises(ValueError):
            Chunking(strategy='unknown')
        with self.assertRaises(ValueError):
            Chunking(max_chars=16001)

    def test_configuration_change_reindexes_and_invalidates_old_citations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'note.md').write_text('# First\nLaunch Friday.\n# Second\nOwner Mina.', encoding='utf-8')
            index = RAGIndex(root/'index.db', Embeddings(api_key=''))
            try:
                for config in (Chunking(), Chunking('fixed'), Chunking('fixed', 40, 10)):
                    old_ids = [row[0] for row in index.connection.execute('SELECT chunk_id FROM chunks')]
                    index.chunking = config
                    self.assertEqual(index.index_directory(root).indexed, 1)
                    self.assertTrue(all(index.source(old_id) is None for old_id in old_ids))
                    with patch('rag.index.parse_document', side_effect=AssertionError('unchanged parse')):
                        self.assertEqual(index.index_directory(root).skipped, 1)
                    version = index.connection.execute('SELECT pipeline_version FROM documents').fetchone()[0]
                    self.assertEqual(version, index.processing_version)
            finally:
                index.close()

    def test_background_writer_uses_selected_configuration(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            root = Path(temporary)
            (root/'note.md').write_text('# First\nLaunch Friday.\n# Second\nOwner Mina.', encoding='utf-8')
            service = RAGService(root, root/'index.db', chunking=Chunking('fixed'))
            try:
                self.assertEqual(service.index_documents().indexed, 1)
                self.assertEqual(service.status()['chunking']['strategy'], 'fixed')
                self.assertEqual(service.status()['chunks'], 1)
                source = service.search('Launch', mode='keyword')[0]
                self.assertIn('→', source['location'])
            finally:
                service.close()


if __name__ == '__main__':
    unittest.main()
