from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .text import split_sentences


def now():
    return datetime.now(timezone.utc).isoformat()


class Reviews:
    """Private human verdicts for answer/evidence pairs; no automatic semantic label."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = RLock()
        self.connection.execute('''CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL, answer_status TEXT NOT NULL,
            answer_text TEXT NOT NULL, sources_json TEXT NOT NULL, experiment TEXT NOT NULL DEFAULT '', verdict TEXT NOT NULL, notes TEXT NOT NULL,
            reviewed_at TEXT)''')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS review_claims (
            id TEXT PRIMARY KEY, review_id TEXT NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
            sentence TEXT NOT NULL, citations_json TEXT NOT NULL, verdict TEXT NOT NULL, notes TEXT NOT NULL,
            reviewed_at TEXT)''')
        columns = {row['name'] for row in self.connection.execute('PRAGMA table_info(reviews)')}
        if 'experiment' not in columns:
            self.connection.execute("ALTER TABLE reviews ADD COLUMN experiment TEXT NOT NULL DEFAULT ''")
        self.connection.commit()

    def close(self):
        with self.lock:
            self.connection.close()

    @staticmethod
    def _sources(sources):
        if not isinstance(sources, list) or len(sources) > 10:
            raise ValueError('Answer sources must contain at most 10 entries')
        clean = []
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError('Invalid answer source')
            item = {key: source.get(key) for key in ('number', 'chunk_id', 'title', 'path', 'location', 'excerpt', 'quote_start', 'quote_end')}
            if not isinstance(item['chunk_id'], str) or not isinstance(item['excerpt'], str) or any(len(str(item[key] or '')) > 2000 for key in ('title', 'path', 'location', 'excerpt')):
                raise ValueError('Invalid answer source fields')
            clean.append(item)
        return clean

    def status(self):
        with self.lock:
            rows = [dict(row) for row in self.connection.execute('SELECT * FROM reviews ORDER BY created_at DESC')]
            experiments = {}
            for row in rows:
                row['sources'] = json.loads(row.pop('sources_json'))
                claims = [dict(claim) for claim in self.connection.execute('SELECT * FROM review_claims WHERE review_id=? ORDER BY rowid', (row['id'],))]
                for claim in claims:
                    claim['citations'] = json.loads(claim.pop('citations_json'))
                row['claims'] = claims
                experiment = row['experiment'] or '미지정'
                grouped = experiments.setdefault(experiment, {'label': experiment, 'reviews': {verdict: 0 for verdict in ('pending', 'supported', 'partial', 'unsupported')}, 'claims': {verdict: 0 for verdict in ('pending', 'supported', 'partial', 'unsupported')}})
                grouped['reviews'][row['verdict']] += 1
                for claim in claims:
                    grouped['claims'][claim['verdict']] += 1
            claims = [claim for row in rows for claim in row['claims']]
            return {'reviews': rows, 'summary': {verdict: sum(row['verdict'] == verdict for row in rows) for verdict in ('pending', 'supported', 'partial', 'unsupported')},
                    'claim_summary': {verdict: sum(claim['verdict'] == verdict for claim in claims) for verdict in ('pending', 'supported', 'partial', 'unsupported')},
                    'experiments': [experiments[label] for label in sorted(experiments)],
                    'privacy': 'Local private review journal; answer text, paths, and notes are not safe to publish without review',
                    'provenance': 'Human-entered verdicts; no automatic semantic judgment'}

    def export(self):
        status = self.status()
        return {'schema': 'personal-document-rag-review-aggregate-v1',
                'reviews': status['summary'], 'claims': status['claim_summary'],
                'provenance': status['provenance'],
                'privacy': 'Aggregate counts only; excludes questions, answers, sources, paths, excerpts, notes, timestamps and reviewer identities',
                'limitations': 'Counts are not semantic evaluation or workplace outcomes. They require enough genuine human review records to interpret.'}

    @staticmethod
    def _claims(text):
        claims = []
        attached = re.sub(r"([.!?。])\s+((?:\[\d+\]\s*)+)", lambda match: f"{match.group(1)}\u2063{match.group(2)}\n", text)
        for sentence in split_sentences(attached):
            sentence = sentence.replace('\u2063', ' ')
            citations = [int(value) for value in re.findall(r'\[(\d+)\]', sentence)]
            if citations:
                claims.append((sentence, sorted(set(citations))))
        if len(claims) > 50:
            raise ValueError('Answer has too many cited claims to review')
        return claims

    def save(self, data):
        if not isinstance(data, dict) or not isinstance(data.get('question'), str) or not isinstance(data.get('answer'), dict):
            raise ValueError('question and answer are required')
        question = data['question'].strip()
        answer = data['answer']
        experiment = data.get('experiment', '')
        status, text = answer.get('status'), answer.get('text')
        if not isinstance(experiment, str) or len(experiment.strip()) > 80 or any(ord(character) < 32 for character in experiment) or not 1 <= len(question) <= 2000 or not isinstance(status, str) or not 1 <= len(status) <= 40 or not isinstance(text, str) or not 1 <= len(text) <= 12000:
            raise ValueError('Invalid question or answer')
        sources = self._sources(answer.get('sources'))
        review_id = uuid4().hex
        with self.lock, self.connection:
            self.connection.execute('INSERT INTO reviews(id,created_at,question,answer_status,answer_text,sources_json,experiment,verdict,notes) VALUES(?,?,?,?,?,?,?,?,?)',
                                    (review_id, now(), question, status, text, json.dumps(sources, ensure_ascii=False), experiment.strip(), 'pending', ''))
            for sentence, citations in self._claims(text):
                self.connection.execute('INSERT INTO review_claims(id,review_id,sentence,citations_json,verdict,notes) VALUES(?,?,?,?,?,?)',
                                        (uuid4().hex, review_id, sentence, json.dumps(citations), 'pending', ''))
        return self.status()

    def verdict(self, data):
        if not isinstance(data, dict) or not isinstance(data.get('id'), str) or data.get('verdict') not in {'supported', 'partial', 'unsupported'} or not isinstance(data.get('notes', ''), str) or len(data.get('notes', '')) > 2000:
            raise ValueError('Invalid review verdict or notes')
        with self.lock, self.connection:
            result = self.connection.execute('UPDATE reviews SET verdict=?,notes=?,reviewed_at=? WHERE id=? AND verdict=?',
                                             (data['verdict'], data.get('notes', ''), now(), data['id'], 'pending'))
            if not result.rowcount:
                raise ValueError('Review is missing or already judged; refresh first')
        return self.status()

    def claim_verdict(self, data):
        if not isinstance(data, dict) or not isinstance(data.get('id'), str) or data.get('verdict') not in {'supported', 'partial', 'unsupported'} or not isinstance(data.get('notes', ''), str) or len(data.get('notes', '')) > 2000:
            raise ValueError('Invalid claim verdict or notes')
        with self.lock, self.connection:
            result = self.connection.execute('UPDATE review_claims SET verdict=?,notes=?,reviewed_at=? WHERE id=? AND verdict=?',
                                             (data['verdict'], data.get('notes', ''), now(), data['id'], 'pending'))
            if not result.rowcount:
                raise ValueError('Claim is missing or already judged; refresh first')
        return self.status()
