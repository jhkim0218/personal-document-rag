from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_server


def request(url: str, method: str = "GET", body: dict | None = None) -> object:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=payload, headers=headers, method=method), timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        server = create_server("data/sample", Path(temporary_directory) / "rag.sqlite3", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            summary = request(f"{base_url}/api/index", "POST")
            assert summary["indexed"] >= 20, summary
            status = request(f"{base_url}/api/status")
            assert status["documents"] >= 20, status
            results = request(f"{base_url}/api/search?q=reciprocal%20rank%20fusion")
            assert results and results[0]["title"] == "02_hybrid_retrieval", results
            answer = request(f"{base_url}/api/ask", "POST", {"question": "RRF는 무엇인가?"})
            assert answer["sources"] and "[1]" in answer["text"], answer
            source = request(f"{base_url}/api/source?id={results[0]['chunk_id']}")
            assert source["path"].endswith("02_hybrid_retrieval.md"), source
        finally:
            server.shutdown()
            server.service.close()
            server.server_close()
    print("WEB SMOKE PASSED")


if __name__ == "__main__":
    main()
