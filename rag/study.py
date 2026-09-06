from __future__ import annotations

import random
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class Study:
    """Local user-entered work trials, never synthetic production measurements."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = RLock()
        self.active_clock = None
        self.connection.execute('''CREATE TABLE IF NOT EXISTS trials (
            id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, task_id TEXT NOT NULL,
            label TEXT NOT NULL, method TEXT NOT NULL, position INTEGER NOT NULL,
            state TEXT NOT NULL, started_at TEXT, finished_at TEXT, elapsed_seconds REAL,
            claimed_success INTEGER, verified INTEGER, corrected INTEGER,
            verified_success INTEGER, failure_type TEXT, notes TEXT,
            retry_of TEXT, replaced_by TEXT)''')
        columns = {row['name'] for row in self.connection.execute('PRAGMA table_info(trials)')}
        for name in ('retry_of', 'replaced_by'):
            if name not in columns:
                self.connection.execute(f'ALTER TABLE trials ADD COLUMN {name} TEXT')
        with self.connection:
            self.connection.execute("UPDATE trials SET state='interrupted', finished_at=? WHERE state='active'", (now(),))

    def close(self):
        with self.lock:
            self.connection.close()

    def status(self):
        with self.lock:
            trials = [dict(row) for row in self.connection.execute('SELECT * FROM trials ORDER BY rowid')]
            return {'trials': trials, 'summary': self._summary(trials),
                    'privacy': 'Local private journal; do not publish task labels or notes without review',
                    'provenance': 'User-entered judgments; not independently verified'}

    @staticmethod
    def _summary(trials):
        """Aggregate only completed self-reports; labels and notes never leave this method."""
        completed = [trial for trial in trials if trial['state'] == 'completed']

        def method_summary(method):
            rows = [trial for trial in completed if trial['method'] == method]
            elapsed = [trial['elapsed_seconds'] for trial in rows if trial['elapsed_seconds'] is not None]
            return {'completed': len(rows), 'verified_successes': sum(trial['verified_success'] == 1 for trial in rows),
                    'verified_success_rate': round(sum(trial['verified_success'] == 1 for trial in rows) / len(rows), 4) if rows else None,
                    'median_elapsed_seconds': round(statistics.median(elapsed), 3) if elapsed else None}

        latest = {}
        for trial in completed:
            latest[trial['task_id'], trial['method']] = trial
        paired = []
        for task_id in {trial['task_id'] for trial in completed}:
            baseline, rag = latest.get((task_id, 'baseline')), latest.get((task_id, 'rag'))
            if baseline and rag:
                paired.append((baseline, rag))
        valid_deltas = [baseline['elapsed_seconds'] - rag['elapsed_seconds'] for baseline, rag in paired
                        if baseline['verified_success'] == rag['verified_success'] == 1
                        and baseline['elapsed_seconds'] is not None and rag['elapsed_seconds'] is not None]
        return {'methods': {method: method_summary(method) for method in ('baseline', 'rag')},
                'completed_pairs': len(paired), 'comparable_verified_pairs': len(valid_deltas),
                'baseline_minus_rag_median_seconds': round(statistics.median(valid_deltas), 3) if valid_deltas else None,
                'rag_faster_pairs': sum(delta > 0 for delta in valid_deltas),
                'baseline_faster_pairs': sum(delta < 0 for delta in valid_deltas),
                'tied_pairs': sum(delta == 0 for delta in valid_deltas)}

    def plan(self, labels):
        if not isinstance(labels, list) or not 2 <= len(labels) <= 30 or any(not isinstance(label, str) or not 1 <= len(label.strip()) <= 300 for label in labels):
            raise ValueError('Provide 2–30 task descriptions, each 1–300 characters; 10–15 recommended')
        labels = [label.strip() for label in labels]
        if len(set(labels)) != len(labels):
            raise ValueError('Task descriptions must be distinct')
        with self.lock:
            if self.connection.execute("SELECT 1 FROM trials WHERE state IN ('pending','active')").fetchone():
                raise ValueError('Finish the current plan before creating another')
            random.SystemRandom().shuffle(labels)
            first_method = random.SystemRandom().choice(('baseline', 'rag'))
            plan_id = uuid4().hex
            with self.connection:
                for number, label in enumerate(labels):
                    task_id = uuid4().hex
                    first = first_method if number % 2 == 0 else ('rag' if first_method == 'baseline' else 'baseline')
                    for offset, method in enumerate((first, 'rag' if first == 'baseline' else 'baseline')):
                        self.connection.execute('INSERT INTO trials(id,plan_id,task_id,label,method,position,state) VALUES(?,?,?,?,?,?,?)',
                            (uuid4().hex, plan_id, task_id, label, method, number*2+offset, 'pending'))
            return self.status()

    def retry_interrupted(self):
        with self.lock:
            if self.active_clock is not None or self.connection.execute("SELECT 1 FROM trials WHERE state='pending'").fetchone():
                raise ValueError('Finish or cancel pending trials before retrying interrupted trials')
            interrupted = self.connection.execute("SELECT * FROM trials WHERE state='interrupted' AND replaced_by IS NULL ORDER BY rowid").fetchall()
            if not interrupted:
                raise ValueError('No interrupted trials to retry')
            with self.connection:
                for trial in interrupted:
                    replacement = uuid4().hex
                    self.connection.execute('''INSERT INTO trials(id,plan_id,task_id,label,method,position,state,retry_of)
                        VALUES(?,?,?,?,?,?,?,?)''',
                        (replacement, trial['plan_id'], trial['task_id'], trial['label'], trial['method'], trial['position'], 'pending', trial['id']))
                    self.connection.execute('UPDATE trials SET replaced_by=? WHERE id=?', (replacement, trial['id']))
            return self.status()

    def cancel(self):
        with self.lock:
            with self.connection:
                result = self.connection.execute("UPDATE trials SET state='cancelled', finished_at=? WHERE state IN ('pending','active')", (now(),))
            self.active_clock = None
            if not result.rowcount:
                raise ValueError('No active or pending trial to cancel')
            return self.status()

    def export(self):
        with self.lock:
            status = self.status()
            return {'schema': 'personal-document-rag-study-aggregate-v1', 'summary': status['summary'],
                    'privacy': 'Aggregate only: task labels, notes, paths, and individual trial records are excluded',
                    'provenance': status['provenance']}

    def start(self):
        with self.lock:
            if self.active_clock is not None:
                raise ValueError('A trial is already active')
            trial = self.connection.execute("SELECT * FROM trials WHERE state='pending' ORDER BY rowid LIMIT 1").fetchone()
            if trial is None:
                raise ValueError('No pending trial; create a plan')
            with self.connection:
                self.connection.execute("UPDATE trials SET state='active', started_at=? WHERE id=?", (now(), trial['id']))
            self.active_clock = (trial['id'], time.monotonic())
            return self.status()

    def finish(self, data):
        if not isinstance(data, dict) or any(type(data.get(key)) is not bool for key in ('success', 'verified', 'corrected')):
            raise ValueError('success, verified and corrected must be booleans')
        failure = data.get('failure_type', 'none')
        notes = data.get('notes', '')
        if failure not in {'none', 'retrieval', 'wrong_answer', 'missing_evidence', 'slow', 'other'} or not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError('Invalid failure type or notes longer than 2000 characters')
        with self.lock:
            if self.active_clock is None or data.get('id') != self.active_clock[0]:
                raise ValueError('Trial is not active; refresh before submitting')
            trial_id, started = self.active_clock
            verified_success = data['success'] and data['verified'] and not data['corrected'] and failure == 'none'
            with self.connection:
                self.connection.execute('''UPDATE trials SET state='completed', finished_at=?, elapsed_seconds=?,
                    claimed_success=?, verified=?, corrected=?, verified_success=?, failure_type=?, notes=? WHERE id=?''',
                    (now(), round(max(0, time.monotonic()-started), 3), data['success'], data['verified'], data['corrected'], verified_success, failure, notes, trial_id))
            self.active_clock = None
            return self.status()
