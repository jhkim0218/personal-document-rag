from __future__ import annotations

import json
import hashlib
import sqlite3
import zipfile
from xml.etree.ElementTree import ParseError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from collections.abc import Callable

from .documents import SUPPORTED_EXTENSIONS, Chunking, content_hash, parse_document, to_chunks
from .embeddings import Embeddings
from .local_models import LocalReranker
from .text import BM25, cosine_similarity, keyword_score, token_overlap, retrieval_tokens


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

    def __init__(self, database_path: str | Path, embeddings: Embeddings | None = None, source_root: str | Path | None = None, pipeline_version: str = "parser-1", chunking: Chunking | None = None, reranker: LocalReranker | None = None, ocr=None, hwp=None):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.embeddings = embeddings or Embeddings()
        self.reranker = reranker
        self.ocr = ocr
        self.hwp = hwp
        self.pipeline_version = pipeline_version
        self.chunking = chunking or Chunking()
        self.source_root = str(Path(source_root).resolve()) if source_root is not None else None
        self.path_filter = None
        self._cache_key = None
        self._cache = None
        self.cache_builds = 0
        self.connection.create_function("path_visible", 1, self._path_visible)
        self._create_schema()

    def _path_visible(self, path: str) -> int:
        return int(self.path_filter(path)) if self.path_filter else 1

    @property
    def processing_version(self) -> str:
        return f"{self.pipeline_version}:chunk-v2:{self.chunking.strategy}:{self.chunking.max_chars}:{self.chunking.overlap_chars}:ocr:{self.ocr.mode if self.ocr else 'off'}:hwp:{self.hwp.mode if self.hwp else 'off'}"

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
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
            CREATE TABLE IF NOT EXISTS file_states (
                path TEXT PRIMARY KEY, state TEXT NOT NULL, error TEXT,
                checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS index_revision (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
            INSERT OR IGNORE INTO index_revision VALUES (1,0);
            CREATE TRIGGER IF NOT EXISTS chunks_revision_insert AFTER INSERT ON chunks BEGIN UPDATE index_revision SET value=value+1 WHERE id=1; END;
            CREATE TRIGGER IF NOT EXISTS chunks_revision_delete AFTER DELETE ON chunks BEGIN UPDATE index_revision SET value=value+1 WHERE id=1; END;
            CREATE TRIGGER IF NOT EXISTS chunks_revision_update AFTER UPDATE ON chunks BEGIN UPDATE index_revision SET value=value+1 WHERE id=1; END;
            CREATE TRIGGER IF NOT EXISTS documents_revision_update AFTER UPDATE OF source_root,path,title,embedding_mode ON documents
                BEGIN UPDATE index_revision SET value=value+1 WHERE id=1; END;
            """
        )
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(documents)")}
        if "embedding_mode" not in columns:
            self.connection.execute("ALTER TABLE documents ADD COLUMN embedding_mode TEXT NOT NULL DEFAULT 'local:hash-256'")
        for name, declaration in {"pipeline_version": "TEXT NOT NULL DEFAULT ''", "mtime_ns": "INTEGER NOT NULL DEFAULT 0", "size_bytes": "INTEGER NOT NULL DEFAULT 0"}.items():
            if name not in columns:
                self.connection.execute(f"ALTER TABLE documents ADD COLUMN {name} {declaration}")
        # Derived duplicate storage was never searched. Keep one authoritative chunk store.
        self.connection.execute("DROP TABLE IF EXISTS chunks_fts")
        self.fts_enabled = False
        self.connection.commit()

    def index_directory(self, directory: str | Path, candidates: list[Path] | None = None, *, force: bool = False, strict: bool = True, prune: bool = True, progress: Callable | None = None, cancelled: Callable | None = None) -> IndexSummary:
        root = Path(directory).resolve()
        if not root.is_dir():
            raise ValueError(f"Index directory does not exist: {root}")
        indexed = skipped = removed = 0
        failed: list[str] = []
        if candidates is None:
            candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
        seen_paths = {str(path.resolve()) for path in candidates}
        existing = {row["path"]: row for row in self.connection.execute("SELECT * FROM documents WHERE source_root = ?", (str(root),))}

        for path in candidates:
            if cancelled and cancelled():
                return IndexSummary(indexed, skipped, removed, tuple(failed))
            resolved = str(path.resolve())
            if progress:
                progress(resolved, "processing", None)
            try:
                previous = existing.get(resolved)
                before = path.stat()
                compatible = previous and previous["embedding_mode"] == self.embeddings.mode and previous["pipeline_version"] == self.processing_version
                unchanged = False
                if not force and compatible:
                    unchanged = previous["mtime_ns"] == before.st_mtime_ns and previous["size_bytes"] == before.st_size if not strict else previous["content_hash"] == content_hash(path)
                if unchanged:
                    skipped += 1
                    with self.connection:
                        self.connection.execute("UPDATE documents SET mtime_ns=?, size_bytes=? WHERE path=?", (before.st_mtime_ns, before.st_size, resolved))
                    self._file_state(resolved, "skipped")
                    if progress:
                        progress(resolved, "skipped", None)
                    continue
                document = parse_document(path, ocr=self.ocr, hwp=self.hwp)
                chunks = to_chunks(document, **asdict(self.chunking))
                vectors = self.embeddings.embed([text for _, text in chunks])
                if len(vectors) != len(chunks) or any(not vector for vector in vectors):
                    raise ValueError("Incomplete embedding response")
                after = path.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size) or document.content_hash != content_hash(path):
                    raise ValueError("File changed while indexing; previous evidence preserved. Retry after saving.")
                if cancelled and cancelled():
                    return IndexSummary(indexed, skipped, removed, tuple(failed))
                with self.connection:
                    self._replace_document(root, document, chunks, vectors)
                    self.connection.execute("UPDATE documents SET pipeline_version=?, mtime_ns=?, size_bytes=? WHERE path=?", (self.processing_version, after.st_mtime_ns, after.st_size, resolved))
                indexed += 1
                self._file_state(resolved, "indexed")
                if progress:
                    progress(resolved, "indexed", None)
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, ParseError, KeyError, sqlite3.Error) as error:
                failed.append(f"{path.name}: {error}")
                self._file_state(resolved, "failed", str(error))
                if progress:
                    progress(resolved, "failed", str(error))

        with self.connection:
            for stored_path, row in existing.items():
                if prune and not (cancelled and cancelled()) and stored_path not in seen_paths:
                    self._delete_document(row["document_id"])
                    self.connection.execute("DELETE FROM file_states WHERE path=?", (stored_path,))
                    removed += 1
        return IndexSummary(indexed=indexed, skipped=skipped, removed=removed, failed=tuple(failed))

    def _file_state(self, path: str, state: str, error: str | None = None) -> None:
        with self.connection:
            self.connection.execute("INSERT INTO file_states(path,state,error) VALUES (?,?,?) ON CONFLICT(path) DO UPDATE SET state=excluded.state,error=excluded.error,checked_at=CURRENT_TIMESTAMP", (path, state, error))

    def _replace_document(self, root: Path, document, chunks, vectors) -> None:
        document_id = hashlib.sha256(f"{document.path.resolve()}:{document.content_hash}:{self.processing_version}".encode("utf-8")).hexdigest()
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

    def _delete_document(self, document_id: str) -> None:
        self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def _search_data(self, use_cache=True):
        revision = self.connection.execute("SELECT value FROM index_revision WHERE id=1").fetchone()[0]
        key = (revision, self.source_root, self.path_filter)
        if not use_cache or self._cache_key != key:
            rows = list(self.connection.execute("SELECT c.*, d.embedding_mode FROM chunks c JOIN documents d USING(document_id) WHERE (? IS NULL OR d.source_root = ?) AND path_visible(c.path)", (self.source_root, self.source_root)))
            data = {"rows": rows, "bm25": BM25([row["text"] for row in rows]), "vectors": None, "enhanced": None}
            self.cache_builds += 1
            if use_cache:
                self._cache_key, self._cache = key, data
            return data
        return self._cache

    def status(self) -> dict[str, object]:
        scope = " WHERE (? IS NULL OR source_root = ?) AND path_visible(documents.path)"
        params = (self.source_root, self.source_root)
        document_count, last_indexed = self.connection.execute("SELECT COUNT(*), MAX(indexed_at) FROM documents" + scope, params).fetchone()
        chunk_count = self.connection.execute("SELECT COUNT(*) FROM chunks JOIN documents USING(document_id)" + scope, params).fetchone()[0]
        return {"documents": document_count, "chunks": chunk_count, "last_indexed_at": last_indexed, "fts_enabled": self.fts_enabled, "embedding_mode": self.embeddings.mode,
                "reranker_mode": f"local:cross-encoder:{self.reranker.name}" if self.reranker else "rules", "ocr_mode": self.ocr.mode if self.ocr else "off", "hwp_mode": self.hwp.mode if self.hwp else "off"}

    def source(self, chunk_id: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT c.chunk_id, c.path, c.title, c.location, c.text, d.content_hash, d.indexed_at, d.mtime_ns FROM chunks c JOIN documents d USING(document_id) WHERE chunk_id = ? AND (? IS NULL OR d.source_root = ?) AND path_visible(c.path)", (chunk_id, self.source_root, self.source_root)
        ).fetchone()
        return dict(row) if row else None

    def document_source(self, path: str) -> dict[str, object] | None:
        row = self.connection.execute('SELECT chunk_id FROM chunks WHERE path=? ORDER BY rowid LIMIT 1', (str(Path(path).resolve()),)).fetchone()
        return self.source(row['chunk_id']) if row else None

    def search(
        self, query: str, limit: int = 5, mode: Literal["vector", "keyword", "hybrid"] = "hybrid", rerank: bool = True, *, use_cache: bool = True, lexical: str = "enhanced"
    ) -> list[SearchResult]:
        if not query.strip() or limit < 1:
            return []
        if mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError("Unknown search mode")
        if lexical not in {"legacy", "enhanced"}:
            raise ValueError("Unknown lexical profile")
        data = self._search_data(use_cache)
        rows = data["rows"]
        if not rows:
            return []
        if mode != "keyword" and any(row["embedding_mode"] != self.embeddings.mode for row in rows):
            raise ValueError("Embedding mode changed. Re-index documents before semantic search.")
        query_embedding = self.embeddings.embed([query])[0] if mode != "keyword" else []
        if mode != "keyword" and data["vectors"] is None:
            data["vectors"] = [json.loads(row["embedding"]) for row in rows]
        vector_scores = {row["chunk_id"]: cosine_similarity(query_embedding, vector) for row, vector in zip(rows, data["vectors"] or [])} if mode != "keyword" else {}
        vector_ranked = sorted(((row, vector_scores[row["chunk_id"]]) for row in rows), key=lambda item: item[1], reverse=True) if mode != "keyword" else []
        bm25 = data["bm25"].scores(query)
        if lexical == "enhanced" and mode != "vector":
            if data["enhanced"] is None:
                data["enhanced"] = BM25([row["title"] + " " + row["path"] + " " + row["text"] for row in rows], retrieval_tokens)
            supplements = data["enhanced"].scores(query)
            bm25 = [base + 0.35 * extra for base, extra in zip(bm25, supplements)]
            needle = query.strip().lower().replace("\\", "/")
            bm25 = [score + (4.0 if needle in row["path"].lower().replace("\\", "/") or needle in row["title"].lower() else 0.0) for score, row in zip(bm25, rows)]
        keyword_ranked = sorted(zip(rows, bm25), key=lambda item: item[1], reverse=True)
        if mode == "vector":
            combined = {row["chunk_id"]: {"row": row, "vector": score, "keyword": keyword_score(query, row["text"]), "score": score} for row, score in vector_ranked[: limit * 4]}
        elif mode == "keyword":
            combined = {row["chunk_id"]: {"row": row, "vector": 0.0, "keyword": score, "score": score} for row, score in keyword_ranked[: limit * 4] if score > 0}
        else:
            combined: dict[str, dict[str, object]] = {}
            for rank, (row, score) in enumerate(vector_ranked[: limit * 4], start=1):
                combined[row["chunk_id"]] = {"row": row, "vector": score, "keyword": keyword_score(query, row["text"]), "score": 1 / (60 + rank)}
            for rank, (row, score) in enumerate(keyword_ranked[: limit * 4], start=1):
                if score <= 0:
                    continue
                item = combined.setdefault(
                    row["chunk_id"],
                    {"row": row, "vector": vector_scores.get(row["chunk_id"], 0.0), "keyword": keyword_score(query, row["text"]), "score": 0.0},
                )
                item["score"] = float(item["score"]) + 1 / (60 + rank)

        local_scores = self.reranker.scores(query, [item["row"]["text"] for item in combined.values()]) if rerank and self.reranker else []
        results = []
        for position, item in enumerate(combined.values()):
            row = item["row"]
            overlap = token_overlap(query, row["text"])
            score = float(item["score"])
            if rerank and self.reranker:
                score = local_scores[position]
            elif rerank:
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
