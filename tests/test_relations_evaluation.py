from __future__ import annotations

import unittest

from scripts.evaluate_relations import ROOT, evaluate


class RelationEvaluationTests(unittest.TestCase):
    def test_explicit_relation_coverage_is_reported_separately_from_search(self):
        report = evaluate(ROOT/'data/eval/relations_cases.json')
        self.assertEqual(report['api_usage']['requests'], 0)
        self.assertEqual(report['metrics']['explicit_relation_document_coverage'], 1.0)
        case = report['cases'][0]
        self.assertEqual(case['relation_coverage'], 1.0)
        self.assertEqual(case['relation_unavailable_documents'], 0)
        self.assertIn('not extracted entities', ' '.join(report['limitations']))


if __name__ == '__main__':
    unittest.main()
