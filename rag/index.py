from __future__ import annotations

import json
import hashlib
import sqlite3
import zipfile
from xml.etree.ElementTree import ParseError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .documents import SUPPORTED_EXTENSIONS, content_hash, parse_document, to_chunks
from .embeddings import Embeddings
from .text import bm25_scores, cosine_similarity, keyword_score, token_overlap


@dataclass(frozen=True)
class IndexSummary:
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    failed: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    path: str
    title: str
    location: str
    text: str
    score: float
    keyword_score: float
    vector_score: float
    overlap: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RAGIndex:
    """SQLite-backed local index. It never changes the source documents."""

    def __init__(self, database_path: str | Path, embeddings: Embeddings | None = None, source_root: str | Path | None = None):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.embeddings = embeddings or Embeddings()
        self.source_root = str(Path(source_root).resolve()) if source_root is not None else None
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_root TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_mode TEXT NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                location TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
            """
        )
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(documents)")}
        if "embedding_mode" not in columns:
            self.connection.execute("ALTER TABLE documents ADD COLUMN embedding_mode TEXT NOT NULL DEFAULT 'local:hash-256'")
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, document_id UNINDEXED, text)"
            )
            self.fts_enabled = True
        except sqlite3.OperationalError:
            self.fts_enabled = False
        self.connection.commit()

    def index_directory(self, directory: str | Path) -> IndexSummary:
        root = Path(directory).resolve()
        if not root.is_dir():
            raise ValueError(f"Index directory does not exist: {root}")
        indexed = skipped = removed = 0
        failed: list[str] = []
        candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
        seen_paths = {str(path.resolve()) for path in candidates}
        existing = {row["path"]: row for row in self.connection.execute("SELECT * FROM documents WHERE source_root = ?", (str(root),))}

        for path in candidates:
            resolved = str(path.resolve())
            try:
                previous = existing.get(resolved)
                if previous and previous["content_hash"] == content_hash(path) and previous["embedding_mode"] == self.embeddings.mode:
                    skipped += 1
                    continue
                document = parse_document(path)
                chunks = to_chunks(document)
                vectors = self.embeddings.embed([text for _, text in chunks])
                if len(vectors) != len(chunks) or any(not vector for vector in vectors):
                    raise ValueError("Incomplete embedding response")
                with self.connection:
                    self._replace_document(root, document, chunks, vectors)
                indexed += 1
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, ParseError, KeyError, sqlite3.Error) as error:
                failed.append(f"{path.name}: {error}")

        with self.connection:
            for stored_path, row in existing.items():
                if stored_path not in seen_paths:
                    self._delete_document(row["document_id"])
                    removed += 1
        return IndexSummary(indexed=indexed, skipped=skipped, removed=removed, failed=tuple(failed))

    def _replace_document(self, root: Path, document, chunks, vectors) -> None:
        document_id = hashlib.sha256(f"{document.path.resolve()}:{document.content_hash}".encode("utf-8")).hexdigest()
        current = self.connection.execute("SELECT document_id FROM documents WHERE path = ?", (str(document.path.resolve()),)).fetchone()
        if current:
            self._delete_document(current["document_id"])
        self.connection.execute(
            "INSERT INTO documents(document_id, source_root, path, title, content_hash, embedding_mode) VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, str(root), str(document.path.resolve()), document.title, document.content_hash, self.embeddings.mode),
        )
        for number, ((location, text), vector) in enumerate(zip(chunks, vectors), start=1):
            chunk_id = f"{document_id}:{number}"
            embedding = json.dumps(vector, separators=(",", ":"))
            self.connection.execute(
                "INSERT INTO chunks(chunk_id, document_id, path, title, location, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chunk_id, document_id, str(document.path.resolve()), document.title, location, text, embedding),
            )
            if self.fts_enabled:
                self.connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, document_id, text) VALUES (?, ?, ?)", (chunk_id, document_id, text)
                )

    def _delete_document(self, document_id: str) -> None:
        if self.fts_enabled:
            self.connection.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
        self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def status(self) -> dict[str, object]:
        scope = " WHERE (? IS NULL OR source_root = ?)"
        params = (self.source_root, self.source_root)
        document_count, last_indexed = self.connection.execute("SELECT COUNT(*), MAX(indexed_at) FROM documents" + scope, params).fetchone()
        chunk_count = self.connection.execute("SELECT COUNT(*) FROM chunks JOIN documents USING(document_id)" + scope, params).fetchone()[0]
        return {"documents": document_count, "chunks": chunk_count, "last_indexed_at": last_indexed, "fts_enabled": self.fts_enabled, "embedding_mode": self.embeddings.mode}

    def source(self, chunk_id: str) -> dict[str, str] | None:
        row = self.connection.execute(
            "SELECT c.chunk_id, c.path, c.title, c.location, c.text, d.content_hash, d.indexed_at FROM chunks c JOIN documents d USING(document_id) WHERE chunk_id = ? AND (? IS NULL OR d.source_root = ?)", (chunk_id, self.source_root, self.source_root)
        ).fetchone()
        return dict(row) if row else None

    def search(
        self, query: str, limit: int = 5, mode: Literal["vector", "keyword", "hybrid"] = "hybrid", rerank: bool = True
    ) -> list[SearchResult]:
        if not query.strip() or limit < 1:
            return []
        if mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError("Unknown search mode")
        rows = list(self.connection.execute("SELECT c.*, d.embedding_mode FROM chunks c JOIN documents d USING(document_id) WHERE (? IS NULL OR d.source_root = ?)", (self.source_root, self.source_root)))
        if not rows:
            return []
        if mode != "keyword" and any(row["embedding_mode"] != self.embeddings.mode for row in rows):
            raise ValueError("Embedding mode changed. Re-index documents before semantic search.")
        query_embedding = self.embeddings.embed([query])[0] if mode != "keyword" else []
        vector_ranked = sorted(
            ((row, cosine_similarity(query_embedding, json.loads(row["embedding"]))) for row in rows), key=lambda item: item[1], reverse=True
        )
        bm25 = bm25_scores(query, [row["text"] for row in rows])
        keyword_ranked = sorted(zip(rows, bm25), key=lambda item: item[1], reverse=True)
        if mode == "vector":
            combined = {row["chunk_id"]: {"row": row, "vector": score, "keyword": keyword_score(query, row["text"]), "score": score} for row, score in vector_ranked[: limit * 4]}
        elif mode == "keyword":
            combined = {row["chunk_id"]: {"row": row, "vector": cosine_similarity(query_embedding, json.loads(row["embedding"])), "keyword": score, "score": score} for row, score in keyword_ranked[: limit * 4] if score > 0}
        else:
            combined: dict[str, dict[str, object]] = {}
            for rank, (row, score) in enumerate(vector_ranked[: limit * 4], start=1):
                combined[row["chunk_id"]] = {"row": row, "vector": score, "keyword": keyword_score(query, row["text"]), "score": 1 / (60 + rank)}
            for rank, (row, score) in enumerate(keyword_ranked[: limit * 4], start=1):
                if score <= 0:
                    continue
                item = combined.setdefault(
                    row["chunk_id"],
                    {"row": row, "vector": cosine_similarity(query_embedding, json.loads(row["embedding"])), "keyword": keyword_score(query, row["text"]), "score": 0.0},
                )
                item["score"] = float(item["score"]) + 1 / (60 + rank)

        results = []
        for item in combined.values():
            row = item["row"]
            overlap = token_overlap(query, row["text"])
            score = float(item["score"])
            if rerank:
                score += overlap * 0.08 + (0.03 if query.lower() in row["text"].lower() else 0.0)
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    path=row["path"],
                    title=row["title"],
                    location=row["location"],
                    text=row["text"],
                    score=score,
                    keyword_score=float(item["keyword"]),
                    vector_score=float(item["vector"]),
                    overlap=overlap,
                )
            )
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]
