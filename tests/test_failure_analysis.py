from __future__ import annotations

import unittest

from scripts.analyze_failures import ROOT, analyze


class FailureAnalysisTests(unittest.TestCase):
    def test_five_reproducible_failures_keep_unknown_investigation_fields_null(self):
        analysis = analyze(ROOT/'results/evaluation.json', ROOT/'data/eval/questions.jsonl')
        self.assertEqual(len(analysis['failures']), 5)
        self.assertTrue(all(item['root_cause'] is None and item['change_applied'] is None for item in analysis['failures']))
        self.assertTrue(all(item['review_state'] == 'needs_human_investigation' for item in analysis['failures']))


if __name__ == '__main__':
    unittest.main()
