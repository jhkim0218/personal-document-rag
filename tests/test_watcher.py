import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.service import RAGService


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.file = self.root/'guide.txt'
        self.file.write_text('Launch Friday.', encoding='utf-8')
        self.service = RAGService(self.root, self.root/'index.db', mode='offline')
        self.watcher = self.service.watcher
        self.watcher.stopped.clear()  # deterministic clock-driven ticks instead of sleeping

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def finish(self, now):
        self.assertNotIn(self.service.jobs.wait(3)['state'], {'running', 'cancelling'})
        self.watcher.tick(now)

    def test_stable_writes_coalesce_and_deletion_is_indexed(self):
        self.watcher.tick(0)
        self.file.write_text('Launch Monday.', encoding='utf-8')
        self.watcher.tick(1)
        self.watcher.tick(2)
        self.assertEqual(self.service.jobs.status()['state'], 'idle')
        self.watcher.tick(3)
        self.finish(4)
        self.assertIn('Monday', self.service.search('Launch')[0]['text'])
        previous = self.service.jobs.status()['id']
        self.watcher.tick(10)
        self.assertEqual(self.service.jobs.status()['id'], previous)
        self.file.unlink()
        self.watcher.tick(11)
        self.watcher.tick(13)
        self.finish(14)
        self.assertEqual(self.service.status()['documents'], 0)

    def test_failed_job_retries_after_backoff(self):
        self.watcher.tick(0)
        with patch.object(self.service.index.embeddings, 'embed', side_effect=ValueError('temporary parse dependency')):
            self.watcher.tick(2)
            self.finish(3)
        self.assertEqual(self.watcher.status()['state'], 'retry_wait')
        previous = self.service.jobs.status()['id']
        self.watcher.tick(32)
        self.assertEqual(self.service.jobs.status()['id'], previous)
        self.watcher.tick(33)
        self.finish(34)
        self.assertEqual(self.service.status()['documents'], 1)

    def test_stop_prevents_work_and_disabled_source_is_not_scanned(self):
        self.service.configure_sources({'sources': [{'path': str(self.root), 'enabled': False}]})
        self.watcher.tick(0)
        self.watcher.tick(2)
        self.finish(3)
        self.assertEqual(self.service.status()['documents'], 0)
        previous = self.service.jobs.status()['id']
        self.watcher.stop()
        self.file.write_text('Launch changed.', encoding='utf-8')
        self.watcher.tick(10)
        self.assertEqual(self.service.jobs.status()['id'], previous)

    def test_change_during_hash_defers_indexing(self):
        from rag.watcher import content_hash
        def changing(path):
            digest = content_hash(path)
            self.file.write_text('Still being saved, incomplete version.', encoding='utf-8')
            return digest
        with patch('rag.watcher.content_hash', side_effect=changing):
            self.watcher.tick(0)
        self.assertEqual(self.service.jobs.status()['state'], 'idle')
        self.assertEqual(self.watcher.status()['state'], 'settling')

    def test_real_worker_stops_promptly_and_does_not_leave_thread(self):
        self.watcher.stopped.set()
        self.watcher.start()
        deadline = time.monotonic()+2
        while self.watcher.status()['state'] == 'starting' and time.monotonic() < deadline:
            time.sleep(.01)
        self.watcher.stop()
        self.assertFalse(self.watcher.thread.is_alive())
        self.assertFalse(self.watcher.status()['enabled'])

    def test_change_before_completion_observation_is_not_swallowed(self):
        self.watcher.tick(0)
        self.watcher.tick(2)
        self.service.jobs.wait(3)
        first_id = self.service.jobs.status()['id']
        self.file.write_text('Launch Tuesday.', encoding='utf-8')
        self.watcher.tick(3)
        self.watcher.tick(5)
        self.finish(6)
        self.assertNotEqual(self.service.jobs.status()['id'], first_id)
        self.assertIn('Tuesday', self.service.search('Launch')[0]['text'])

    def test_cancelled_automatic_job_pauses_watching(self):
        self.watcher.active_job = 'watch-job'
        with patch.object(self.service.jobs, 'status', return_value={'id': 'watch-job', 'state': 'cancelled'}):
            self.watcher.tick(0)
        self.assertFalse(self.watcher.status()['enabled'])
        self.assertEqual(self.watcher.status()['state'], 'paused_after_cancel')
