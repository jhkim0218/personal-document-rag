import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.compare_chunking import ROOT, compare, evidence_hits, validate_labels


class ChunkingExperimentTests(unittest.TestCase):
    def test_gold_document_with_wrong_sentence_does_not_count(self):
        root = Path.cwd()
        gold = [{'document': 'gold.md', 'text': 'The launch is Friday.'}]
        rows = [SimpleNamespace(path=str(root/'gold.md'), text='The owner is Mina.')]
        self.assertEqual(evidence_hits(gold, rows, root), [False])
        rows[0].text = gold[0]['text']
        self.assertEqual(evidence_hits(gold, rows, root), [True])
        rows[0].path = str(root/'wrong.md')
        self.assertEqual(evidence_hits(gold, rows, root), [False])

    def test_invalid_source_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'gold.md').write_text('# Decision\nLaunch Friday.', encoding='utf-8')
            fixture = {'documents': {'gold.md': ''}, 'queries': [{'id': 'q', 'question': 'When?', 'gold': [
                {'document': 'gold.md', 'location': 'heading: Decision', 'text': 'Launch Monday.'}]}]}
            with self.assertRaises(ValueError):
                validate_labels(fixture, root)
            fixture['queries'][0]['gold'][0]['text'] = 'Launch Friday.'
            validate_labels(fixture, root)

    def test_short_windows_record_misses_without_treating_them_as_harness_errors(self):
        report = compare(ROOT/'data/eval/chunking_cases.json', 300, 40)
        misses = [reason for variant in report['variants'] for case in variant['cases']
                  for reason in case['miss_reasons'] if reason]
        self.assertTrue(misses)
        self.assertTrue(all(reason in {'ranking', 'split_across_chunks'} for reason in misses))

    def test_controlled_offline_experiment_records_both_strategies(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'must-not-be-used'}):
            report = compare(ROOT/'data/eval/chunking_cases.json')
        self.assertEqual(report['api_usage']['requests'], 0)
        self.assertEqual([v['chunking']['strategy'] for v in report['variants']], ['fixed', 'structured'])
        self.assertTrue(all(length > 900 for length in report['corpus_characters'].values()))
        for variant in report['variants']:
            cases = variant['cases']
            self.assertEqual(len(cases), 6)
            self.assertEqual(sum(len(c['hits']) for c in cases), 8)
            self.assertEqual(variant['micro_literal_evidence_recall_at_5'], sum(sum(c['hits']) for c in cases)/8)
            self.assertTrue(all('chunk_id' not in gold for c in cases for gold in c['gold']))
        self.assertNotEqual(report['variants'][0]['chunks'], report['variants'][1]['chunks'])


if __name__ == '__main__':
    unittest.main()
