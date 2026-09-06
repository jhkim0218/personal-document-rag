from __future__ import annotations

import unittest

from scripts.validate_holdout import validate


class HoldoutValidationTests(unittest.TestCase):
    def test_human_reviewed_holdout_requires_evidence_fields_and_no_dev_overlap(self):
        development = [{'id': 'dev', 'question': 'What is the launch date?'}]
        holdout = [{'id': 'holdout', 'question': 'What rollback approval is required?', 'answerable': True,
                    'expected_document': 'runbook.md', 'gold_location': 'heading: Rollback',
                    'required_facts': ['approval'], 'label_status': 'human-reviewed'},
                   {'id': 'unknown', 'question': 'What is the fictional planet?', 'answerable': False,
                    'label_status': 'human-reviewed'}]
        report = validate(development, holdout)
        self.assertTrue(report['valid'])
        self.assertEqual((report['human_reviewed'], report['answerable'], report['unanswerable']), (2, 1, 1))

    def test_drafts_and_duplicate_or_development_questions_cannot_be_final_holdout_evidence(self):
        development = [{'id': 'dev', 'question': 'What is the launch date?'}]
        holdout = [{'id': 'dev', 'question': ' What  is  the launch date? ', 'answerable': True,
                    'expected_document': '', 'gold_location': '', 'required_facts': [], 'label_status': 'ai-draft'}]
        report = validate(development, holdout)
        self.assertFalse(report['valid'])
        self.assertEqual(report['ai_draft'], 1)
        self.assertTrue(any('overlaps' in error for error in report['errors']))

    def test_empty_holdout_is_not_evidence(self):
        report = validate([], [])
        self.assertFalse(report['valid'])
        self.assertIn('holdout must contain at least one case', report['errors'])
