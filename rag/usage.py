from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock


class APIUsageJournal:
    """Persistent, content-free API attempt records for one local user."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = Lock()
        self.error = None
        self.connection.execute('''CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, time_utc TEXT NOT NULL, model TEXT NOT NULL, endpoint TEXT NOT NULL,
            attempt INTEGER NOT NULL, status TEXT, elapsed_ms REAL, usage_json TEXT, estimated_cost_usd REAL, pricing_json TEXT)''')
        if 'pricing_json' not in {row[1] for row in self.connection.execute('PRAGMA table_info(attempts)')}:
            self.connection.execute('ALTER TABLE attempts ADD COLUMN pricing_json TEXT')
        self.connection.commit()

    def record(self, record):
        safe = {key: record.get(key) for key in ('time_utc', 'model', 'endpoint', 'attempt', 'status', 'elapsed_ms', 'usage', 'estimated_cost_usd', 'pricing')}
        try:
            with self.lock, self.connection:
                self.connection.execute('INSERT INTO attempts(time_utc,model,endpoint,attempt,status,elapsed_ms,usage_json,estimated_cost_usd,pricing_json) VALUES(?,?,?,?,?,?,?,?,?)',
                                        (safe['time_utc'], safe['model'], safe['endpoint'], safe['attempt'], str(safe['status']) if safe['status'] is not None else None,
                                         safe['elapsed_ms'], json.dumps(safe['usage'], ensure_ascii=False) if safe['usage'] else None, safe['estimated_cost_usd'],
                                         json.dumps(safe['pricing'], ensure_ascii=False) if safe['pricing'] else None))
        except (OSError, sqlite3.Error) as error:
            self.error = str(error)

    def status(self):
        with self.lock:
            rows = [dict(row) for row in self.connection.execute('SELECT time_utc,model,endpoint,attempt,status,elapsed_ms,usage_json,estimated_cost_usd,pricing_json FROM attempts ORDER BY id DESC LIMIT 100')]
        for row in rows:
            row['usage'] = json.loads(row.pop('usage_json')) if row['usage_json'] else None
            row['pricing'] = json.loads(row.pop('pricing_json')) if row['pricing_json'] else None
        return {'records': rows, 'persistent': True, 'last_error': self.error,
                'privacy': 'Stores model, endpoint, attempt, status, elapsed time and token usage only; never request/response content or keys'}

    def close(self):
        with self.lock:
            self.connection.close()
