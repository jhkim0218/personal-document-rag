from __future__ import annotations

import json
import time
from threading import Event, Lock, Thread

from .documents import content_hash


class FolderWatcher:
    """Single-user polling watcher; reuse the existing safe indexing worker."""

    def __init__(self, service):
        self.service = service
        self.stopped = Event()
        self.stopped.set()
        self.control = Lock()
        self.thread = None
        self.record = {'state': 'stopped', 'error': None}
        self.candidate = self.baseline = self.planned = None
        self.active_job = None
        self.stable_since = self.retry_at = 0

    def status(self):
        return {**self.record, 'enabled': not self.stopped.is_set(), 'poll_seconds': 5, 'settle_seconds': 2, 'retry_seconds': 30}

    def start(self):
        with self.control:
            if self.thread and self.thread.is_alive():
                return self.status()
            self.candidate = self.baseline = self.planned = None
            self.active_job = None
            self.stable_since = self.retry_at = 0
            self.stopped.clear()
            self.record = {'state': 'starting', 'error': None}
            self.thread = Thread(target=self._run, daemon=True)
            self.thread.start()
            return self.status()

    def stop(self):
        with self.control:
            self.stopped.set()
            if self.thread:
                self.thread.join()
            self.record = {'state': 'stopped', 'error': None}
            return self.status()

    def _run(self):
        while not self.stopped.is_set():
            try:
                self.tick(time.monotonic())
            except Exception as error:
                self.record = {'state': 'retry_wait', 'error': f'{type(error).__name__}: {error}'}
                self.retry_at = time.monotonic() + 30
            self.stopped.wait(5)

    def tick(self, now):
        if self.stopped.is_set():
            return
        job = self.service.jobs.status()
        if job['state'] in {'running', 'cancelling'}:
            self.record = {'state': 'indexing', 'error': None}
            return
        if self.active_job:
            if job.get('id') == self.active_job and job['state'] == 'succeeded':
                self.baseline = self.planned
            elif job.get('id') == self.active_job and job['state'] == 'cancelled':
                self.stopped.set()
                self.record = {'state': 'paused_after_cancel', 'error': 'Restart watching to resume automatic indexing'}
                return
            else:
                self.retry_at = now + 30
                self.record = {'state': 'retry_wait', 'error': job.get('error') or 'Index incomplete; retry scheduled'}
            self.active_job = None
        if now < self.retry_at:
            return
        with self.service.lock:
            settings = self.service.settings
            version = self.service.index.processing_version
        files = sorted({path for source in settings.sources for path in settings.files(source)})
        snapshot = []
        # ponytail: full hashes every poll detect same-mtime edits; OS events can replace polling if IO becomes a measured bottleneck.
        for path in files:
            if self.stopped.is_set():
                return
            before = path.stat()
            digest = content_hash(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                self.candidate = None
                self.record = {'state': 'settling', 'error': None}
                return
            snapshot.append((str(path), after.st_size, after.st_mtime_ns, digest))
        snapshot = (json.dumps(settings.as_dict(), sort_keys=True), version, tuple(snapshot))
        if snapshot != self.candidate:
            self.candidate, self.stable_since = snapshot, now
            self.record = {'state': 'settling', 'error': None}
            return
        if now - self.stable_since < 2 or snapshot == self.baseline:
            self.record = {'state': 'watching', 'error': None}
            return
        with self.service.lock:
            if self.stopped.is_set() or settings is not self.service.settings:
                return
            started = self.service.start_index()
        self.active_job, self.planned = started['id'], snapshot
        self.record = {'state': 'indexing', 'error': None}
