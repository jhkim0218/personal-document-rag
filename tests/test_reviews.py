from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.reviews import Reviews


def answer():
    return {'status': 'fallback', 'text': 'Launch is Friday. [1]', 'sources': [{'number': 1, 'chunk_id': 'chunk-1', 'title': 'Decision', 'path': 'C:/private/decision.md', 'location': 'heading: Launch', 'excerpt': 'Launch is Friday.', 'context': 'Do not persist this full context'}]}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)/'reviews.sqlite3'
        self.reviews = Reviews(self.path)

    def tearDown(self):
        self.reviews.close()
        self.temporary.cleanup()

    def test_pending_review_sanitizes_sources_and_allows_one_human_verdict(self):
        saved = self.reviews.save({'question': 'When is launch?', 'answer': answer()})
        review = saved['reviews'][0]
        self.assertEqual(review['verdict'], 'pending')
        self.assertNotIn('context', review['sources'][0])
        self.assertEqual(self.reviews.verdict({'id': review['id'], 'verdict': 'supported', 'notes': 'Opened original.'})['summary']['supported'], 1)
        with self.assertRaises(ValueError):
            self.reviews.verdict({'id': review['id'], 'verdict': 'unsupported'})

    def test_private_records_persist_on_restart(self):
        self.reviews.save({'question': 'Private?', 'answer': answer()})
        self.reviews.close()
        self.reviews = Reviews(self.path)
        self.assertEqual(len(self.reviews.status()['reviews']), 1)
        self.assertIn('Human-entered', self.reviews.status()['provenance'])

    def test_cited_claims_are_separate_private_human_review_units(self):
        saved = self.reviews.save({'question': 'When is launch?', 'answer': answer()})
        claim = saved['reviews'][0]['claims'][0]
        self.assertEqual(claim['sentence'], 'Launch is Friday. [1]')
        self.assertEqual(claim['citations'], [1])
        judged = self.reviews.claim_verdict({'id': claim['id'], 'verdict': 'supported', 'notes': 'Literal source checked'})
        self.assertEqual(judged['claim_summary']['supported'], 1)
        with self.assertRaises(ValueError):
            self.reviews.claim_verdict({'id': claim['id'], 'verdict': 'unsupported'})

    def test_aggregate_export_excludes_private_review_content(self):
        self.reviews.save({'question': 'Private launch question?', 'answer': answer()})
        exported = self.reviews.export()
        self.assertEqual(exported['schema'], 'personal-document-rag-review-aggregate-v1')
        self.assertEqual(exported['reviews']['pending'], 1)
        self.assertEqual(exported['claims']['pending'], 1)
        self.assertNotIn('Private launch question', json.dumps(exported))
        self.assertNotIn('C:/private', json.dumps(exported))

    def test_private_experiment_groups_compare_verdicts_without_public_labels(self):
        baseline = self.reviews.save({'question': 'Private baseline question?', 'answer': answer(), 'experiment': 'Private baseline'})['reviews'][0]
        candidate = self.reviews.save({'question': 'Private candidate question?', 'answer': answer(), 'experiment': 'Private candidate'})['reviews'][0]
        status = self.reviews.verdict({'id': baseline['id'], 'verdict': 'supported'})
        groups = {group['label']: group for group in status['experiments']}
        self.assertEqual(groups['Private baseline']['reviews']['supported'], 1)
        self.assertEqual(groups['Private candidate']['reviews']['pending'], 1)
        exported = json.dumps(self.reviews.export())
        self.assertNotIn('Private baseline', exported)
        self.assertNotIn('Private candidate', exported)
