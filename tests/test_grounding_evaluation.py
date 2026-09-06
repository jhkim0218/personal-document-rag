from __future__ import annotations

import unittest

from scripts.evaluate_grounding import ROOT, evaluate, literal_cited_claim_support, source_matches_gold


class GroundingEvaluationTests(unittest.TestCase):
    def test_same_document_wrong_excerpt_and_false_cited_claim_fail(self):
        root = ROOT / 'data/eval'
        gold = {'document': 'decision.md', 'text': 'Launch is Friday.'}
        source = {'path': str(root/'decision.md'), 'excerpt': 'The owner is Mina.', 'context': 'The owner is Mina.'}
        self.assertFalse(source_matches_gold(source, gold, root))
        self.assertEqual(literal_cited_claim_support('Launch is Friday. [1]', [{'number': 1, 'context': 'Launch is Monday.'}]), (0, 1))
        self.assertEqual(literal_cited_claim_support('Launch is Friday. [1]', [{'number': 1, 'context': 'Launch is Friday.'}]), (1, 1))

    def test_offline_grounding_report_keeps_semantic_metrics_null(self):
        report = evaluate(ROOT/'data/eval/chunking_cases.json')
        self.assertEqual(report['api_usage']['requests'], 0)
        self.assertEqual(report['metrics']['semantic_citation_precision'], None)
        self.assertEqual(report['metrics']['grounded_answer_rate'], None)
        self.assertEqual(len(report['cases']), 6)
        self.assertIn('literal_gold_evidence_recall_at_5', report['metrics'])
