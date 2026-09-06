from __future__ import annotations

import unittest

from scripts.evaluate_answer_policy import ROOT, evaluate


class AnswerPolicyEvaluationTests(unittest.TestCase):
    def test_explicit_missing_requirements_are_partial_only_under_required_policy(self):
        report = evaluate(ROOT/'data/eval/answer_policy_cases.json')
        self.assertEqual(report['api_usage']['requests'], 0)
        self.assertEqual(report['metrics']['required_literal_partial_rate'], 1.0)
        self.assertEqual(report['metrics']['permissive_literal_partial_rate'], 0.0)
        complete = next(case for case in report['cases'] if case['id'] == 'complete-change')
        self.assertNotEqual(complete['statuses']['required_literal_partial'], 'partial')


if __name__ == '__main__':
    unittest.main()
