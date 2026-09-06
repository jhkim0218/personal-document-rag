from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock, Thread
from uuid import uuid4

from .index import IndexSummary, RAGIndex


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexJobs:
    """One indexing worker; its SQLite connection is never shared with readers."""

    def __init__(self, database: Path):
        self.database = database
        self.path = database.with_suffix(".jobs.json")
        self.lock = RLock()
        self.cancelled = Event()
        self.thread = None
        self.record = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"state": "idle", "last_success_at": None}
        if self.record["state"] in {"running", "cancelling"}:
            self.record.update(state="interrupted", error="Process stopped before completion. Run changed-file indexing again.", finished_at=now())
            self._save()

    def _save(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def status(self):
        with self.lock:
            return json.loads(json.dumps(self.record))

    def start(self, settings, embeddings, pipeline_version, *, mode="changed", strict=True, chunking=None):
        if mode not in {"changed", "full", "retry"} or not isinstance(strict, bool):
            raise ValueError("mode must be changed/full/retry and strict must be boolean")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Indexing is already running")
            retry_paths = {item["path"] for item in self.record.get("failures", [])}
            if mode == "retry" and not retry_paths:
                raise ValueError("No failed files to retry; run changed-file indexing")
            self.record = {"id": uuid4().hex, "state": "running", "mode": mode, "strict": strict,
                           "started_at": now(), "finished_at": None, "last_success_at": self.record.get("last_success_at"),
                           "current_file": None, "completed": 0, "total": 0, "failures": [], "error": None,
                           "summary": asdict(IndexSummary())}
            self.cancelled.clear()
            self._save()
            self.thread = Thread(target=self._run, args=(settings, embeddings, pipeline_version, retry_paths, chunking), daemon=True)
            self.thread.start()
            return self.status()

    def cancel(self):
        with self.lock:
            if self.record["state"] == "running":
                self.cancelled.set()
                self.record["state"] = "cancelling"
                self._save()
            return self.status()

    def wait(self, timeout=None):
        if self.thread:
            self.thread.join(timeout)
        return self.status()

    def _progress(self, path, state, error):
        with self.lock:
            self.record["current_file"] = path
            if state != "processing":
                self.record["completed"] += 1
            if state in {"indexed", "skipped"}:
                self.record["summary"][state] += 1
            if state == "failed":
                self.record["failures"].append({"path": path, "error": error})
            self._save()

    def _run(self, settings, embeddings, pipeline_version, retry_paths, chunking):
        index = None
        summaries = []
        try:
            plan = []
            owned = set()
            # Overlapping roots own each file once, in configured order.
            for source in settings.sources:
                if not source.enabled:
                    continue
                files = settings.files(source)
                selected = [p for p in files if str(p.resolve()) not in owned]
                owned.update(str(p.resolve()) for p in selected)
                if self.record["mode"] == "retry":
                    selected = [p for p in selected if str(p.resolve()) in retry_paths]
                plan.append((source, selected))
            with self.lock:
                self.record["total"] = sum(len(files) for _, files in plan)
                self._save()
            index = RAGIndex(self.database, embeddings=embeddings, pipeline_version=pipeline_version, chunking=chunking)
            for source, files in plan:
                if self.cancelled.is_set():
                    break
                summaries.append(index.index_directory(source.path, files, force=self.record["mode"] == "full",
                    strict=self.record["strict"], prune=self.record["mode"] != "retry", progress=self._progress,
                    cancelled=self.cancelled.is_set))
            with self.lock:
                self.record["summary"] = asdict(IndexSummary(sum(s.indexed for s in summaries), sum(s.skipped for s in summaries),
                    sum(s.removed for s in summaries), tuple(e for s in summaries for e in s.failed)))
                if self.record["mode"] == "retry":
                    missing = retry_paths - owned
                    self.record["failures"].extend({"path": path, "error": "Failed file is missing or outside active scope; run changed-file indexing"} for path in sorted(missing))
                self.record["state"] = "cancelled" if self.cancelled.is_set() else "partial" if self.record["failures"] else "succeeded"
                if self.record["state"] == "succeeded":
                    self.record["last_success_at"] = now()
        except Exception as error:
            # Worker boundary: preserve and surface even unexpected parser failures, never silently lose a thread.
            with self.lock:
                self.record.update(state="failed", error=f"{type(error).__name__}: {error}")
        finally:
            if index:
                index.close()
            with self.lock:
                self.record.update(finished_at=now(), current_file=None)
                self._save()
