from __future__ import annotations

import argparse
import json
import hashlib
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rag.service import RAGService
from rag.documents import Chunking




def make_handler(service: RAGService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # Keep normal UI use quiet.
            return

        def do_GET(self) -> None:
            try:
                self._get()
            except (OSError, ValueError, RuntimeError) as error:
                self._send_json({"error": str(error), "kind": "input_error"}, 400)
            except Exception:
                self._send_json({"error": "Unexpected server error. Check the operation status and retry.", "kind": "server_error"}, 500)

        def _get(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(Path(__file__).with_name("web").joinpath("index.html").read_text(encoding="utf-8"))
            elif parsed.path == "/request.js":
                encoded = Path(__file__).with_name("web").joinpath("request.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            elif parsed.path == "/api/status":
                self._send_json(service.status())
            elif parsed.path == "/api/study":
                self._send_json(service.study.status())
            elif parsed.path == "/api/study/export":
                self._send_json(service.study.export())
            elif parsed.path == "/api/reviews":
                self._send_json(service.reviews.status())
            elif parsed.path == "/api/reviews/export":
                self._send_json(service.reviews.export())
            elif parsed.path == "/api/relations":
                self._send_json(service.relations_view(parse_qs(parsed.query).get('q', [''])[0]))
            elif parsed.path == "/api/index/job":
                self._send_json(service.jobs.status())
            elif parsed.path == "/api/sources":
                self._send_json(service.status()["settings"])
            elif parsed.path == "/api/sources/preview":
                self._send_json(service.preview_sources())
            elif parsed.path == "/api/search":
                query = parse_qs(parsed.query).get("q", [""])[0]
                self._send_json(service.search(query))
            elif parsed.path == "/api/history":
                query = parse_qs(parsed.query).get("q", [""])[0]
                self._send_json(service.history(query))
            elif parsed.path == "/api/bundle":
                query = parse_qs(parsed.query).get("q", [""])[0]
                self._send_json(service.bundle(query))
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
                    content = escape(source["text"])
                    offsets = parse_qs(parsed.query)
                    if "start" in offsets or "end" in offsets:
                        start = int(offsets.get("start", ["-1"])[0])
                        end = int(offsets.get("end", ["-1"])[0])
                        if not 0 <= start < end <= len(source["text"]):
                            raise ValueError("Invalid evidence span")
                        content = escape(source["text"][:start]) + '<mark id="quote">' + escape(source["text"][start:end]) + '</mark>' + escape(source["text"][end:])
                    body = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>근거 확인</title><style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:1rem}pre{white-space:pre-wrap;line-height:1.8;background:#f4f6f8;padding:1rem}</style>'
                    body += f'<a href="/">검색으로 돌아가기</a><h1>{escape(source["title"])}</h1><p>{escape(source["path"])}</p><p>{escape(source["location"])}</p><p>색인 당시 본문 · {escape(source["indexed_at"])}</p><pre id="evidence">{content}</pre>'
                    page = source["location"].split(" ·")[0].removeprefix("page ")
                    fragment = f"#page={page}" if page.isdigit() else ""
                    body += f'<a href="/api/file?id={source["chunk_id"]}{fragment}">원본 파일 확인</a></html>'
                    self._send_html(body)
                else:
                    path = Path(source["path"]).resolve()
                    if not service.allows_file(path) or not path.is_file():
                        self._send_json({"error": "Original file is no longer available in the active folder"}, 404)
                        return
                    encoded = path.read_bytes()
                    if hashlib.sha256(encoded).hexdigest() != source["content_hash"]:
                        self._send_json({"error": "Original file changed. Re-index before opening this citation."}, 409)
                        return
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
                origin = self.headers.get("Origin")
                if origin and origin != "http://" + self.headers.get("Host", ""):
                    self._send_json({"error": "Cross-origin writes are not allowed"}, 403)
                    return
                self._post()
            except RuntimeError as error:
                self._send_json({"error": str(error), "kind": "conflict"}, 409)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                self._send_json({"error": str(error), "kind": "input_error"}, 400)
            except Exception:
                self._send_json({"error": "Unexpected server error. Check the operation status and retry.", "kind": "server_error"}, 500)

        def _post(self) -> None:
            if self.path in {"/api/study/start", "/api/study/retry", "/api/study/cancel"}:
                action = {"/api/study/start": service.study.start, "/api/study/retry": service.study.retry_interrupted,
                          "/api/study/cancel": service.study.cancel}[self.path]
                self._send_json(action())
                return
            if self.path in {"/api/study/plan", "/api/study/finish"}:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be between 1 and 65536 bytes")
                body = json.loads(self.rfile.read(length))
                result = service.study.plan(body.get('tasks')) if self.path.endswith('/plan') else service.study.finish(body)
                self._send_json(result)
                return
            if self.path in {"/api/reviews/save", "/api/reviews/verdict", "/api/reviews/claim-verdict"}:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be between 1 and 65536 bytes")
                body = json.loads(self.rfile.read(length))
                result = service.reviews.save(body) if self.path.endswith('/save') else service.reviews.claim_verdict(body) if self.path.endswith('/claim-verdict') else service.reviews.verdict(body)
                self._send_json(result)
                return
            if self.path == "/api/relations":
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Relations require application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be between 1 and 65536 bytes")
                self._send_json(service.configure_relations(json.loads(self.rfile.read(length))))
                return
            if self.path == "/api/watch/start":
                self._send_json(service.watcher.start())
                return
            if self.path == "/api/watch/stop":
                self._send_json(service.watcher.stop())
                return
            if self.path == "/api/index/cancel":
                self._send_json(service.jobs.cancel())
                return
            if self.path == "/api/index/start":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be between 1 and 65536 bytes")
                body = json.loads(self.rfile.read(length))
                self._send_json(service.start_index(body.get("mode", "changed"), body.get("strict", True)), 202)
                return
            if self.path in {"/api/sources", "/api/sources/preview"}:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Source settings require application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must be between 1 and 65536 bytes")
                body = json.loads(self.rfile.read(length))
                result = service.configure_sources(body) if self.path == "/api/sources" else service.preview_sources(body)
                self._send_json(result)
                return
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


def create_server(source_directory: str | Path, database_path: str | Path, port: int = 8000, chunking: Chunking | None = None, mode: str = "auto", local_embedding_model: str | None = None, local_reranker_model: str | None = None, ocr: bool = False, ocr_executable: str = 'tesseract', ocr_language: str = 'eng', hwp_executable: str | None = None) -> ThreadingHTTPServer:
    service = RAGService(source_directory, database_path, chunking=chunking, mode=mode, local_embedding_model=local_embedding_model, local_reranker_model=local_reranker_model, ocr=ocr, ocr_executable=ocr_executable, ocr_language=ocr_language, hwp_executable=hwp_executable)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    server.service = service  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Local evidence-first personal document RAG")
    parser.add_argument("--data", default="data/sample", help="Folder containing documents to index")
    parser.add_argument("--db", default=".local/rag.sqlite3", help="Local SQLite index path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["auto", "offline", "local"], default="auto", help="offline ignores API keys; local uses only supplied local model directories; auto uses configured API keys")
    parser.add_argument("--local-embedding-model", help="Existing local sentence-transformers model directory; local mode never downloads a model")
    parser.add_argument("--local-reranker-model", help="Existing local cross-encoder model directory; local mode never downloads a model")
    parser.add_argument("--ocr", action="store_true", help="Run local Tesseract on image-only PDF pages")
    parser.add_argument("--ocr-executable", default="tesseract", help="Existing local Tesseract executable")
    parser.add_argument("--ocr-language", default="eng", help="Installed Tesseract language code")
    parser.add_argument("--hwp-executable", help="Existing local hwp5txt-compatible HWP v5 converter")
    parser.add_argument("--open-browser", action="store_true", help="Open the local page in the default browser")
    parser.add_argument("--watch", action="store_true", help="Automatically index stable changes in active folders")
    parser.add_argument("--chunk-strategy", choices=["structured", "fixed"], default="structured")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=160)
    args = parser.parse_args()
    server = create_server(args.data, args.db, args.port, Chunking(args.chunk_strategy, args.chunk_size, args.chunk_overlap), mode=args.mode,
                           local_embedding_model=args.local_embedding_model, local_reranker_model=args.local_reranker_model,
                           ocr=args.ocr, ocr_executable=args.ocr_executable, ocr_language=args.ocr_language, hwp_executable=args.hwp_executable)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Open {url} (source: {Path(args.data).resolve()})")
    try:
        status = server.service.status()
        if args.watch:
            server.service.watcher.start()
        print(f"Mode: {status['runtime_mode']} | search: {status['embedding_mode']} | generation: {status['generation_mode']} | external transmission: {status['external_transmission']}")
        if args.open_browser:
            try:
                if not webbrowser.open(url):
                    print(f"Browser did not open. Open {url} manually.")
            except (webbrowser.Error, OSError) as error:
                print(f"Browser launch failed: {error}. Open {url} manually.")
        server.serve_forever()
    finally:
        server.service.close()  # type: ignore[attr-defined]
        server.server_close()


if __name__ == "__main__":
    main()
