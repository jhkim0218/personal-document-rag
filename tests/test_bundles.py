from __future__ import annotations

import unittest

from rag.bundles import related_bundle
from rag.index import SearchResult


def result(chunk_id: str, path: str) -> SearchResult:
    return SearchResult(chunk_id, path, path.rsplit('/', 1)[-1], 'heading: Evidence', f'{chunk_id} evidence', 1, 1, 1, 1)


class BundleTests(unittest.TestCase):
    def test_bundle_keeps_the_best_ranked_chunk_per_document(self):
        bundle = related_bundle('rollback decision', [result('a1', 'C:/spec.md'), result('a2', 'C:/spec.md'), result('b1', 'C:/minutes.md'), result('c1', 'C:/runbook.md')], limit=3)
        self.assertEqual([source['chunk_id'] for source in bundle['sources']], ['a1', 'b1', 'c1'])
        self.assertEqual([source['retrieval_rank'] for source in bundle['sources']], [1, 3, 4])
        self.assertEqual(bundle['candidate_chunks'], 4)
        self.assertEqual(bundle['candidate_documents'], 3)
        self.assertEqual(bundle['documents_selected'], 3)
