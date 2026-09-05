from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Literal

from .answer import Answer, Answerer
from .index import IndexSummary, RAGIndex


class RAGService:
    def __init__(self, source_directory: str | Path, database_path: str | Path, answerer: Answerer | None = None):
        self.source_directory = Path(source_directory).resolve()
        self.index = RAGIndex(database_path, source_root=self.source_directory)
        self.answerer = answerer or Answerer()
        # ponytail: serialize DB access for one user; per-request connections if concurrency becomes important.
        self.lock = RLock()
        self.last_index_result = None

    def close(self) -> None:
        with self.lock:
            self.index.close()

    def index_documents(self) -> IndexSummary:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another request is using the index. Please retry shortly.")
        try:
            self.last_index_result = self.index.index_directory(self.source_directory)
            return self.last_index_result
        finally:
            self.lock.release()

    def status(self) -> dict[str, object]:
        with self.lock:
            return {**self.index.status(), "source_directory": str(self.source_directory)}

    def search(
        self, query: str, limit: int = 5, mode: Literal["vector", "keyword", "hybrid"] = "hybrid", rerank: bool = True
    ) -> list[dict[str, object]]:
        with self.lock:
            return [result.as_dict() for result in self.index.search(query, limit=limit, mode=mode, rerank=rerank)]

    def ask(self, question: str, limit: int = 5) -> Answer:
        with self.lock:
            results = self.index.search(question, limit=limit, mode="hybrid", rerank=True)
        return self.answerer.answer(question, results)

    def source(self, chunk_id: str) -> dict[str, str] | None:
        with self.lock:
            return self.index.source(chunk_id)
