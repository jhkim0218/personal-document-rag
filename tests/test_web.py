from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path

from app import create_server


class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.source = Path(self.temporary.name) / "documents"
        self.source.mkdir()
        (self.source / "guide.md").write_text("# Retrieval\nRRF combines keyword and vector rankings.", encoding="utf-8")
        self.server = create_server(self.source, Path(self.temporary.name) / "index.sqlite3", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.service.close()
        self.server.server_close()
        self.temporary.cleanup()

    def request(self, route: str, method: str = "GET", body: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        with urllib.request.urlopen(urllib.request.Request(self.base + route, data=data, headers=headers, method=method)) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_http_routes_index_search_ask_and_source(self) -> None:
        self.assertEqual(self.request("/api/index", "POST")["indexed"], 1)
        self.assertEqual(self.request("/api/status")["documents"], 1)
        results = self.request("/api/search?q=keyword")
        self.assertEqual(results[0]["title"], "guide")
        answer = self.request("/api/ask", "POST", {"question": "What does RRF combine?"})
        self.assertIn("[1]", answer["text"])
        source = self.request("/api/source?id=" + results[0]["chunk_id"])
        self.assertTrue(source["path"].endswith("guide.md"))

    def test_evidence_snapshot_and_original_file_consistency(self) -> None:
        document = self.source / "guide.md"
        original = "# Retrieval\nRRF <script>alert(1)</script> " + "evidence " * 100
        document.write_text(original, encoding="utf-8")
        self.request("/api/index", "POST")
        chunk_id = self.request("/api/search?q=RRF")[0]["chunk_id"]
        with urllib.request.urlopen(self.base + "/source?id=" + chunk_id) as response:
            page = response.read().decode("utf-8")
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertIn("evidence " * 80, page)
        with urllib.request.urlopen(self.base + "/api/file?id=" + chunk_id) as response:
            self.assertEqual(response.read(), document.read_bytes())
        document.write_text("Changed after indexing", encoding="utf-8")
        with self.assertRaises(urllib.error.HTTPError) as changed:
            urllib.request.urlopen(self.base + "/api/file?id=" + chunk_id)
        self.assertEqual(changed.exception.code, 409)
        for route in ("/source?id=unknown", "/api/file?id=unknown"):
            with self.assertRaises(urllib.error.HTTPError) as missing:
                urllib.request.urlopen(self.base + route)
            self.assertEqual(missing.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
