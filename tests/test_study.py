import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.study import Study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)/'study.sqlite3'
        self.study = Study(self.path)

    def tearDown(self):
        self.study.close()
        self.temporary.cleanup()

    def test_pairs_are_counterbalanced_and_empty_history_is_not_results(self):
        self.assertEqual(self.study.status()['trials'], [])
        rows = self.study.plan([f'Task {i}' for i in range(10)])['trials']
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(t['method']=='rag' for t in rows[::2]), 5)
        for first, second in zip(rows[::2], rows[1::2]):
            self.assertEqual(first['task_id'], second['task_id'])
            self.assertNotEqual(first['method'], second['method'])
            self.assertIsNone(first['elapsed_seconds'])
        with self.assertRaises(ValueError):
            self.study.plan(['new', 'other'])

    def test_elapsed_time_and_unverified_or_corrected_results(self):
        self.study.plan(['Find decision', 'Find runbook'])
        for verified, corrected, expected in ((True, False, 1), (False, False, 0), (True, True, 0), (True, False, 0)):
            with patch('rag.study.time.monotonic', return_value=100):
                active = next(t for t in self.study.start()['trials'] if t['state']=='active')
            with self.assertRaises(ValueError):
                self.study.start()
            data = {'id':active['id'], 'success':True, 'verified':verified, 'corrected':corrected,
                    'failure_type':'wrong_answer' if verified and not corrected and expected == 0 else 'none'}
            with patch('rag.study.time.monotonic', return_value=112.5):
                completed = next(t for t in self.study.finish(data)['trials'] if t['id']==active['id'])
            self.assertEqual(completed['elapsed_seconds'], 12.5)
            self.assertEqual(completed['verified_success'], expected)
            with self.assertRaises(ValueError):
                self.study.finish(data)

    def test_restart_preserves_private_records_but_interrupts_unfinished_timer(self):
        self.study.plan(['Private task', 'Second task'])
        self.study.start()
        self.study.close()
        self.study = Study(self.path)
        rows = self.study.status()['trials']
        interrupted = next(t for t in rows if t['state']=='interrupted')
        self.assertIsNone(interrupted['elapsed_seconds'])
        self.assertIsNone(interrupted['verified_success'])
        self.assertTrue(any(t['state']=='active' for t in self.study.start()['trials']))

    def test_bad_judgments_and_duplicate_tasks_rejected(self):
        with self.assertRaises(ValueError):
            self.study.plan(['same', 'same'])
        with self.assertRaises(ValueError):
            self.study.finish({'success':'true', 'verified':True, 'corrected':False})

    def test_interrupted_trial_is_reassigned_without_inventing_measurement(self):
        self.study.plan(['Private task', 'Second task'])
        self.study.start()
        self.study.close()
        self.study = Study(self.path)
        with self.assertRaises(ValueError):
            self.study.retry_interrupted()
        self.study.cancel()
        retried = self.study.retry_interrupted()['trials']
        original = next(t for t in retried if t['state'] == 'interrupted')
        replacement = next(t for t in retried if t['retry_of'] == original['id'])
        self.assertEqual(original['replaced_by'], replacement['id'])
        self.assertEqual(replacement['state'], 'pending')
        self.assertIsNone(original['elapsed_seconds'])
        self.assertIsNone(original['verified_success'])

    def test_private_aggregate_excludes_labels_and_uses_only_verified_pairs(self):
        self.study.plan(['Secret launch deadline', 'Private incident note'])
        for _ in range(4):
            with patch('rag.study.time.monotonic', return_value=100):
                active = next(t for t in self.study.start()['trials'] if t['state'] == 'active')
            elapsed = 20 if active['method'] == 'baseline' else 10
            with patch('rag.study.time.monotonic', return_value=100 + elapsed):
                self.study.finish({'id': active['id'], 'success': True, 'verified': True, 'corrected': False})
        exported = self.study.export()
        encoded = str(exported)
        self.assertNotIn('Secret launch deadline', encoded)
        self.assertNotIn('Private incident note', encoded)
        self.assertEqual(exported['summary']['completed_pairs'], 2)
        self.assertEqual(exported['summary']['comparable_verified_pairs'], 2)
        self.assertEqual(exported['summary']['baseline_minus_rag_median_seconds'], 10)
