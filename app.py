from __future__ import annotations

import argparse
import json
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rag.service import RAGService
from rag.documents import content_hash




def make_handler(service: RAGService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # Keep normal UI use quiet.
            return

        def do_GET(self) -> None:
            try:
                self._get()
            except (OSError, ValueError, RuntimeError) as error:
                self._send_json({"error": str(error)}, 400)

        def _get(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(Path(__file__).with_name("web").joinpath("index.html").read_text(encoding="utf-8"))
            elif parsed.path == "/api/status":
                self._send_json(service.status())
            elif parsed.path == "/api/search":
                query = parse_qs(parsed.query).get("q", [""])[0]
                self._send_json(service.search(query))
            elif parsed.path == "/api/source":
                chunk_id = parse_qs(parsed.query).get("id", [""])[0]
                source = service.source(chunk_id)
                self._send_json(source or {"error": "Source not found"}, 200 if source else 404)
            elif parsed.path in {"/source", "/api/file"}:
                source = service.source(parse_qs(parsed.query).get("id", [""])[0])
                if not source:
                    self._send_json({"error": "Source not found in the active folder"}, 404)
                    return
                if parsed.path == "/source":
                    body = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>근거 확인</title><style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:1rem}pre{white-space:pre-wrap;line-height:1.8;background:#f4f6f8;padding:1rem}</style>'
                    body += f'<a href="/">검색으로 돌아가기</a><h1>{escape(source["title"])}</h1><p>{escape(source["path"])}</p><p>{escape(source["location"])}</p><p>색인 당시 본문 · {escape(source["indexed_at"])}</p><pre id="evidence">{escape(source["text"])}</pre>'
                    page = source["location"].split(" ·")[0].removeprefix("page ")
                    fragment = f"#page={page}" if page.isdigit() else ""
                    body += f'<a href="/api/file?id={source["chunk_id"]}{fragment}">원본 파일 확인</a></html>'
                    self._send_html(body)
                else:
                    path = Path(source["path"]).resolve()
                    if not path.is_relative_to(service.source_directory) or not path.is_file():
                        self._send_json({"error": "Original file is no longer available in the active folder"}, 404)
                        return
                    if content_hash(path) != source["content_hash"]:
                        self._send_json({"error": "Original file changed. Re-index before opening this citation."}, 409)
                        return
                    encoded = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream")
                    if path.suffix.lower() != ".pdf":
                        self.send_header("Content-Disposition", 'attachment; filename="document' + path.suffix.lower() + '"')
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
            else:
                self._send_json({"error": "Not found"}, 404)

        def do_POST(self) -> None:
            try:
                self._post()
            except RuntimeError as error:
                self._send_json({"error": str(error)}, 409)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                self._send_json({"error": str(error)}, 400)

        def _post(self) -> None:
            if self.path == "/api/index":
                summary = service.index_documents()
                self._send_json({"indexed": summary.indexed, "skipped": summary.skipped, "removed": summary.removed, "failed": list(summary.failed)})
                return
            if self.path == "/api/ask":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 65536:
                        raise ValueError("Request body must be between 1 and 65536 bytes")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    question = str(body.get("question", "")).strip()
                    if not question:
                        raise ValueError("question is required")
                    self._send_json(service.ask(question).as_dict())
                except (ValueError, json.JSONDecodeError) as error:
                    self._send_json({"error": str(error)}, 400)
                return
            self._send_json({"error": "Not found"}, 404)

        def _send_json(self, body: object, status: int = 200) -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def create_server(source_directory: str | Path, database_path: str | Path, port: int = 8000) -> ThreadingHTTPServer:
    service = RAGService(source_directory, database_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    server.service = service  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Local evidence-first personal document RAG")
    parser.add_argument("--data", default="data/sample", help="Folder containing documents to index")
    parser.add_argument("--db", default=".local/rag.sqlite3", help="Local SQLite index path")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = create_server(args.data, args.db, args.port)
    print(f"Open http://127.0.0.1:{args.port} (source: {Path(args.data).resolve()})")
    try:
        server.serve_forever()
    finally:
        server.service.close()  # type: ignore[attr-defined]
        server.server_close()


if __name__ == "__main__":
    main()
