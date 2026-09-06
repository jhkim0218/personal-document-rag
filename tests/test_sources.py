from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.service import RAGService
from rag.sources import SourceSettings
from tests import test_web


class SourceTests(unittest.TestCase):
    def test_multiple_roots_filters_disable_and_restart_share_one_scope(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            root = Path(temporary)
            a, b = root / "a", root / "b"
            a.mkdir()
            b.mkdir()
            (a / "alpha.txt").write_text("alpha launch", encoding="utf-8")
            (b / "beta.txt").write_text("beta launch", encoding="utf-8")
            for folder in (".git", ".local", "excluded"):
                (a / folder).mkdir()
                (a / folder / "hidden.txt").write_text("hidden launch", encoding="utf-8")
            (a / "~$temporary.txt").write_text("temporary launch", encoding="utf-8")
            settings = {"sources": [{"path": str(a), "excludes": ["excluded"]}, {"path": str(b)}], "extensions": [".txt"]}
            service = RAGService(a, root / "index.db")
            try:
                service.configure_sources(settings)
                self.assertEqual(service.preview_sources()["count"], 2)
                self.assertEqual(service.index_documents().indexed, 2)
                results = service.search("launch", mode="keyword")
                self.assertEqual(len(results), 2)
                beta = next(r for r in results if r["title"] == "beta")
                self.assertTrue(service.allows_file(Path(beta["path"])))
                settings["sources"][1]["enabled"] = False
                service.configure_sources(settings)
                self.assertEqual(service.status()["documents"], 1)
                self.assertEqual(service.preview_sources()["count"], 1)
                self.assertEqual([r["title"] for r in service.search("launch", mode="keyword")], ["alpha"])
                self.assertIsNone(service.source(beta["chunk_id"]))
                self.assertFalse(service.allows_file(Path(beta["path"])))
            finally:
                service.close()
            service = RAGService(a, root / "index.db")
            try:
                self.assertEqual(service.status()["documents"], 1)
                self.assertFalse(service.settings.sources[1].enabled)
                service.configure_sources({"sources": [], "extensions": [".txt"]})
                self.assertEqual(service.status()["documents"], 0)
                self.assertEqual(service.search("launch"), [])
            finally:
                service.close()

    def test_include_extension_and_parent_escape_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selected").mkdir()
            for path in (root / "outside.txt", root / "selected" / "yes.txt", root / "selected" / "no.md"):
                path.write_text("document", encoding="utf-8")
            settings = SourceSettings.from_dict({"sources": [{"path": temporary, "includes": ["selected"]}], "extensions": [".txt"]})
            self.assertEqual(settings.preview()["files"], [str((root / "selected" / "yes.txt").resolve())])
            with self.assertRaises(ValueError):
                SourceSettings.from_dict({"sources": [{"path": temporary, "includes": ["../outside"]}]})
            with self.assertRaises(ValueError):
                SourceSettings.from_dict({"sources": [{"path": temporary, "enabled": "false"}]})


class SourceHTTPTests(unittest.TestCase):
    setUp = test_web.WebTests.setUp
    tearDown = test_web.WebTests.tearDown
    request = test_web.WebTests.request
    def test_configure_preview_and_disable_http_evidence(self):
        self.request("/api/index", "POST")
        chunk_id = self.request("/api/search?q=keyword")[0]["chunk_id"]
        preview = self.request("/api/sources/preview", "POST", {"sources": [{"path": str(self.source)}]})
        self.assertEqual(preview["count"], 1)
        self.request("/api/sources", "POST", {"sources": []})
        self.assertEqual(self.request("/api/status")["documents"], 0)
        import urllib.error
        import urllib.request
        for endpoint in ("/source?id=", "/api/file?id=", "/api/source?id="):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(self.base + endpoint + chunk_id)
            self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
