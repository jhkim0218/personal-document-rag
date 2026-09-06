from __future__ import annotations

from pathlib import Path
import json
from dataclasses import asdict
from threading import RLock
from typing import Literal

from .answer import Answer, Answerer
from .index import IndexSummary, RAGIndex
from .sources import SourceSettings
from .jobs import IndexJobs
from .documents import Chunking
from .embeddings import Embeddings
from .watcher import FolderWatcher
from .study import Study
from .history import decision_history
from .bundles import related_bundle
from .reviews import Reviews
from .usage import APIUsageJournal
from .relations import Relations
from .local_models import LocalReranker
from .ocr import TesseractOCR


class RAGService:
    def __init__(self, source_directory: str | Path, database_path: str | Path, answerer: Answerer | None = None, chunking: Chunking | None = None, mode: str = "auto", local_embedding_model: str | None = None, local_reranker_model: str | None = None, ocr: bool = False, ocr_executable: str = 'tesseract', ocr_language: str = 'eng'):
        if mode not in {"auto", "offline", "local"}:
            raise ValueError("mode must be auto or offline")
        if mode in {"offline", "local"} and answerer is not None:
            raise ValueError("Offline and local modes require the built-in local answerer")
        if mode == "local" and not (local_embedding_model or local_reranker_model):
            raise ValueError("Local mode requires a local embedding or reranker model directory")
        if mode != "local" and (local_embedding_model or local_reranker_model):
            raise ValueError("Local model directories require --mode local")
        self.mode = mode
        self.source_directory = Path(source_directory).resolve()
        self.settings_path = Path(database_path).with_suffix(".sources.json")
        data = json.loads(self.settings_path.read_text(encoding="utf-8")) if self.settings_path.exists() else {"sources": [{"path": str(self.source_directory)}]}
        self.settings = SourceSettings.from_dict(data, require_existing=not self.settings_path.exists())
        ocr_engine = TesseractOCR(ocr_executable, ocr_language) if ocr else None
        self.index = RAGIndex(database_path, embeddings=Embeddings(api_key="" if mode in {"offline", "local"} else None, local_model_path=local_embedding_model), chunking=chunking,
                              reranker=LocalReranker(local_reranker_model) if local_reranker_model else None, ocr=ocr_engine)
        self.index.path_filter = self.settings.allows
        self.answerer = answerer or Answerer(api_key="" if mode in {"offline", "local"} else None)
        # ponytail: serialize readers for one user; indexing owns a separate WAL connection.
        self.lock = RLock()
        self.last_index_result = None
        self.jobs = IndexJobs(self.index.database_path)
        self.watcher = FolderWatcher(self)
        self.study = Study(self.index.database_path.with_suffix('.study.sqlite3'))
        self.reviews = Reviews(self.index.database_path.with_suffix('.reviews.sqlite3'))
        self.usage = APIUsageJournal(self.index.database_path.with_suffix('.usage.sqlite3'))
        self.relations = Relations(self.index.database_path.with_suffix('.relations.json'))
        self.index.embeddings.requests.journal = self.usage
        self.answerer.requests.journal = self.usage

    def close(self) -> None:
        self.watcher.stop()
        self.jobs.cancel()
        self.jobs.wait()
        with self.lock:
            self.index.close()
            self.study.close()
            self.reviews.close()
            self.usage.close()

    def index_documents(self) -> IndexSummary:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another request is using the index. Please retry shortly.")
        try:
            self.start_index()
            job = self.jobs.wait()
            if job["state"] == "failed":
                raise RuntimeError(job["error"])
            self.last_index_result = IndexSummary(**job["summary"])
            return self.last_index_result
        finally:
            self.lock.release()

    def start_index(self, mode="changed", strict=True) -> dict:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another request is using the index. Please retry shortly.")
        try:
            return self.jobs.start(self.settings, self.index.embeddings, self.index.pipeline_version, mode=mode, strict=strict, chunking=self.index.chunking, ocr=self.index.ocr)
        finally:
            self.lock.release()

    def status(self) -> dict[str, object]:
        with self.lock:
            return {**self.index.status(), "source_directory": str(self.source_directory), "settings": self.settings.as_dict(), "job": self.jobs.status(), "chunking": asdict(self.index.chunking),
                    "runtime_mode": self.mode,
                    "watcher": self.watcher.status(),
                    "api_attempts": {"embeddings": self.index.embeddings.requests.snapshot(), "generation": self.answerer.requests.snapshot()},
                    "api_usage": self.usage.status(),
                    "generation_mode": f"openai:{self.answerer.model}" if self.answerer.api_key else "local:extractive",
                    "external_transmission": bool(self.index.embeddings.api_key or self.answerer.api_key)}

    def configure_sources(self, data: dict) -> dict:
        settings = SourceSettings.from_dict(data)
        with self.lock:
            if self.jobs.status()["state"] in {"running", "cancelling"}:
                raise RuntimeError("Stop indexing before changing source settings")
            settings.save(self.settings_path)
            self.settings = settings
            self.index.path_filter = settings.allows
            return settings.as_dict()

    def preview_sources(self, data: dict | None = None) -> dict:
        with self.lock:
            return (SourceSettings.from_dict(data) if data is not None else self.settings).preview()

    def allows_file(self, path: Path) -> bool:
        with self.lock:
            return self.settings.allows(path)

    def search(
        self, query: str, limit: int = 5, mode: Literal["vector", "keyword", "hybrid"] = "hybrid", rerank: bool = True
    ) -> list[dict[str, object]]:
        with self.lock:
            return [result.as_dict() for result in self.index.search(query, limit=limit, mode=mode, rerank=rerank)]

    def ask(self, question: str, limit: int = 5) -> Answer:
        with self.lock:
            results = self.index.search(question, limit=limit, mode="hybrid", rerank=True)
        return self.answerer.answer(question, results)

    def history(self, query: str) -> dict[str, object]:
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        with self.lock:
            return decision_history(query, self.index.search(query, limit=10, mode="hybrid", rerank=True))

    def bundle(self, query: str) -> dict[str, object]:
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        with self.lock:
            return related_bundle(query, self.index.search(query, limit=20, mode="hybrid", rerank=True))

    def source(self, chunk_id: str) -> dict[str, str] | None:
        with self.lock:
            return self.index.source(chunk_id)

    def _relation_source(self, path: str):
        source = self.index.document_source(path)
        return source if source and self.settings.allows(source['path']) else None

    def configure_relations(self, data: dict) -> dict:
        with self.lock:
            self.relations.configure(data, self._relation_source)
            return self.relations.view('', self._relation_source)

    def relations_view(self, query: str = '') -> dict:
        with self.lock:
            return self.relations.view(query, self._relation_source)
