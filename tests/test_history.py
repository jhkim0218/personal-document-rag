from __future__ import annotations

import unittest

from rag.history import decision_history
from rag.index import SearchResult


def result(chunk_id: str, path: str, text: str) -> SearchResult:
    return SearchResult(chunk_id, path, path.rsplit('/', 1)[-1], 'heading: Decisions', text, 1, 1, 1, 1)


class HistoryTests(unittest.TestCase):
    def test_explicit_change_is_date_sorted_with_literal_before_after_reason_and_source(self):
        later = result('later', 'C:/docs/later.md', '2026-04-02: Rollout 2026-04-10 → 2026-04-17 변경. 이유: 승인 지연.')
        earlier = result('earlier', 'C:/docs/earlier.md', '2026-03-01: Rollout 2026-03-01 → 2026-03-15 changed because load test failed.')
        history = decision_history('rollout change', [later, earlier])
        self.assertEqual([event['date'] for event in history['events']], ['2026-03-01', '2026-04-02'])
        first = history['events'][0]
        self.assertEqual(first['before'], 'Rollout 2026-03-01')
        self.assertEqual(first['after'], '2026-03-15')
        self.assertEqual(first['reason'], 'load test failed')
        self.assertEqual(first['source']['chunk_id'], 'earlier')
        self.assertEqual(history['events'][1]['reason'], '승인 지연')

    def test_missing_date_values_and_reason_remain_null_instead_of_inference(self):
        history = decision_history('decision', [result('one', 'C:/docs/decision.md', 'The rollout changed after review.'), result('two', 'C:/docs/other.md', 'A dated status exists on 2026-03-01.')])
        self.assertEqual(len(history['events']), 1)
        self.assertIsNone(history['events'][0]['date'])
        self.assertIsNone(history['events'][0]['before'])
        self.assertIsNone(history['events'][0]['after'])
        self.assertIsNone(history['events'][0]['reason'])

    def test_korean_reason_particle_is_not_inferred_into_the_reason_value(self):
        history = decision_history('배포 변경', [result('one', 'C:/docs/decision.md', '배포일을 9월 10일에서 9월 17일로 변경한다. 변경 이유는 승인 지연이다.')])
        self.assertEqual(history['events'][0]['before'], '9월 10일')
        self.assertEqual(history['events'][0]['after'], '9월 17일')
        self.assertEqual(history['events'][0]['reason'], '승인 지연이다')
        self.assertEqual(history['events'][0]['reason_statement'], '변경 이유는 승인 지연이다.')
