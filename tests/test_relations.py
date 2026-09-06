from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.service import RAGService


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/'docs'
        self.root.mkdir()
        self.minutes = self.root/'minutes.md'
        self.spec = self.root/'spec.md'
        self.minutes.write_text('# Launch\nDecision meeting record.', encoding='utf-8')
        self.spec.write_text('# Launch\nApproved rollout specification.', encoding='utf-8')
        self.db = Path(self.temporary.name)/'index.sqlite3'
        self.service = RAGService(self.root, self.db, mode='offline')
        self.service.index_documents()

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def test_explicit_project_decision_documents_persist_and_respect_active_scope(self):
        data = {'projects': [{'name': 'Aurora', 'decisions': [{'name': 'Launch approval', 'documents': [str(self.minutes), str(self.spec)]}]}]}
        view = self.service.configure_relations(data)
        self.assertEqual([document['title'] for document in view['projects'][0]['decisions'][0]['documents']], ['minutes', 'spec'])
        self.assertEqual(len(self.service.relations_view('approval')['projects']), 1)
        self.service.configure_sources({'sources': [{'path': str(self.root), 'enabled': False}], 'extensions': ['.md']})
        hidden = self.service.relations_view('')
        self.assertEqual(hidden['projects'][0]['decisions'][0]['documents'], [])
        self.assertEqual(hidden['unavailable_documents'], 2)

    def test_unindexed_or_out_of_scope_documents_are_rejected(self):
        with self.assertRaises(ValueError):
            self.service.configure_relations({'projects': [{'name': 'Aurora', 'decisions': [{'name': 'Missing', 'documents': [str(self.root/'not-indexed.md')]}]}]})
