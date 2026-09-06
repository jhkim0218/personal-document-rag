from __future__ import annotations

import os
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.embeddings import Embeddings
from rag.index import RAGIndex
from rag.service import RAGService


class IndexJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "docs"
        self.source.mkdir()
        self.file = self.source / "launch.txt"
        self.file.write_text("Launch Friday", encoding="utf-8")
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": ""})
        self.environment.start()
        self.service = RAGService(self.source, self.root / "index.db")

    def tearDown(self):
        self.service.close()
        self.environment.stop()
        self.temporary.cleanup()

    def test_version_change_rebuilds_and_metadata_mode_is_explicit(self):
        index = self.service.index
        index.index_directory(self.source)
        first_id = index.search("Launch")[0].chunk_id
        with patch("rag.index.parse_document", side_effect=AssertionError("unchanged reparse")), patch.object(index.embeddings, "embed", side_effect=AssertionError("unchanged embedding")):
            self.assertEqual(index.index_directory(self.source).skipped, 1)
        old_stat = self.file.stat()
        self.file.write_text("Launch Monday", encoding="utf-8")  # same length and deliberately restored mtime
        os.utime(self.file, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        self.assertEqual(index.index_directory(self.source, strict=False).skipped, 1)
        self.assertEqual(index.index_directory(self.source, strict=True).indexed, 1)
        second_id = index.search("Launch")[0].chunk_id
        self.assertNotEqual(first_id, second_id)
        index.pipeline_version = "parser-2:structured-900-160"
        self.assertEqual(index.index_directory(self.source).indexed, 1)
        self.assertIsNone(index.source(second_id), "Old citation must not point to a newly segmented chunk")
        self.assertEqual(index.index_directory(self.source, force=True).indexed, 1)

    def test_saving_during_embedding_preserves_old_evidence(self):
        index = self.service.index
        index.index_directory(self.source)
        old = index.search("Launch")[0]
        self.file.write_text("Launch Monday", encoding="utf-8")
        original_embed = index.embeddings.embed
        def saving(texts):
            self.file.write_text("Launch Tuesday", encoding="utf-8")
            return original_embed(texts)
        with patch.object(index.embeddings, "embed", side_effect=saving):
            result = index.index_directory(self.source)
        self.assertEqual(len(result.failed), 1)
        self.assertEqual(index.source(old.chunk_id)["text"], "Launch Friday")
        self.assertEqual(index.connection.execute("SELECT state FROM file_states WHERE path=?", (str(self.file),)).fetchone()[0], "failed")

    def test_background_progress_search_duplicate_cancel_and_resume(self):
        self.service.index_documents()
        self.file.write_text("Launch Monday", encoding="utf-8")
        entered, release = threading.Event(), threading.Event()
        original_embed = self.service.index.embeddings.embed
        def delayed(texts):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test worker was not released")
            return original_embed(texts)
        with patch.object(self.service.index.embeddings, "embed", side_effect=delayed):
            try:
                self.service.start_index()
                self.assertTrue(entered.wait(2))
                job = self.service.jobs.status()
                self.assertEqual(job["current_file"], str(self.file))
                self.assertEqual(job["total"], 1)
                self.assertEqual(self.service.search("Launch", mode="keyword")[0]["text"], "Launch Friday")
                self.assertEqual(self.service.status()["documents"], 1)
                with self.assertRaises(RuntimeError):
                    self.service.start_index()
                with self.assertRaises(RuntimeError):
                    self.service.configure_sources({"sources": []})
                self.service.jobs.cancel()
            finally:
                release.set()
                self.service.jobs.wait(3)
        self.assertEqual(self.service.jobs.status()["state"], "cancelled")
        self.assertEqual(self.service.search("Launch", mode="keyword")[0]["text"], "Launch Friday")
        self.assertEqual(self.service.index_documents().indexed, 1)

    def test_corrupt_files_retry_only_failures_and_persist_success_time(self):
        broken = self.source / "broken.docx"
        broken.write_bytes(b"not a zip")
        self.service.index_documents()
        failed = self.service.jobs.status()
        self.assertEqual(failed["state"], "partial")
        self.assertEqual(failed["failures"][0]["path"], str(broken))
        from tests.test_documents_and_index import write_docx
        write_docx(broken, ["Retention thirty days"])
        from rag.index import parse_document
        with patch("rag.index.parse_document", wraps=parse_document) as parser:
            self.service.start_index(mode="retry")
            job = self.service.jobs.wait(3)
        self.assertEqual(parser.call_count, 1)
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(self.service.status()["documents"], 2, "Retry must never delete unrelated healthy documents")
        self.assertIsNotNone(job["last_success_at"])
        previous_time = job["last_success_at"]
        self.service.close()
        self.service = RAGService(self.source, self.root / "index.db")
        self.assertEqual(self.service.jobs.status()["last_success_at"], previous_time)

    def test_corrupt_pdf_and_embedding_timeout_are_file_failures(self):
        self.service.index_documents()
        previous = self.service.search("Launch", mode="keyword")[0]
        self.file.write_text("Launch Monday", encoding="utf-8")
        (self.source / "broken.pdf").write_bytes(b"not a PDF")
        with patch.object(self.service.index.embeddings, "embed", side_effect=TimeoutError("embedding request timed out")):
            self.service.start_index()
            job = self.service.jobs.wait(3)
        self.assertEqual(job["state"], "partial")
        self.assertEqual(len(job["failures"]), 2)
        self.assertEqual(self.service.source(previous["chunk_id"])["text"], "Launch Friday")
        self.assertTrue(any("timed out" in f["error"] for f in job["failures"]))
        self.assertTrue(any("PDF parsing failed" in f["error"] for f in job["failures"]))

    def test_missing_persisted_root_can_be_disabled_without_losing_index(self):
        self.service.configure_sources({"sources": [{"path": str(self.source)}]})
        self.service.index_documents()
        self.service.close()
        self.source.rename(self.root / "temporarily-moved")
        self.service = RAGService(self.source, self.root / "index.db")
        self.service.start_index()
        job = self.service.jobs.wait(3)
        self.assertEqual(job["state"], "failed")
        self.assertIn("unavailable", job["error"])
        self.assertEqual(self.service.status()["documents"], 1)
        self.service.configure_sources({"sources": [{"path": str(self.source), "enabled": False}]})
        self.assertEqual(self.service.status()["documents"], 0)

    def test_interrupted_job_is_not_reported_as_live_after_restart(self):
        self.service.close()
        state = self.root / "index.jobs.json"
        state.write_text(json.dumps({"state": "running", "last_success_at": None}), encoding="utf-8")
        self.service = RAGService(self.source, self.root / "index.db")
        self.assertEqual(self.service.jobs.status()["state"], "interrupted")
        self.assertEqual(self.service.index_documents().indexed, 1)


if __name__ == "__main__":
    unittest.main()
