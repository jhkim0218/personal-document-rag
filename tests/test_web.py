from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch

from app import create_server
from tests.test_documents_and_index import write_pdf


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
        status = self.request("/api/status")
        self.assertEqual(status["documents"], 1)
        self.assertEqual(status["reranker_mode"], "rules")
        self.assertEqual(status["ocr_mode"], "off")
        self.assertEqual(status["hwp_mode"], "off")
        results = self.request("/api/search?q=keyword")
        self.assertEqual(results[0]["title"], "guide")
        answer = self.request("/api/ask", "POST", {"question": "What does RRF combine?"})
        self.assertIn("[1]", answer["text"])
        source = self.request("/api/source?id=" + results[0]["chunk_id"])
        self.assertTrue(source["path"].endswith("guide.md"))
        self.assertEqual(source['mtime_ns'], (self.source/'guide.md').stat().st_mtime_ns)

    def test_request_deadline_script_is_served_as_javascript(self) -> None:
        with urllib.request.urlopen(self.base + "/request.js") as response:
            self.assertEqual(response.headers.get_content_type(), "text/javascript")
            self.assertEqual(response.read(), Path(__file__).resolve().parents[1].joinpath("web/request.js").read_bytes())
        with urllib.request.urlopen(self.base + "/") as response:
            self.assertIn('<script src="/request.js"></script>', response.read().decode("utf-8"))

    def test_watch_start_stop_http_routes(self) -> None:
        self.assertTrue(self.request('/api/watch/start', 'POST')['enabled'])
        self.assertTrue(self.request('/api/status')['watcher']['enabled'])
        self.assertFalse(self.request('/api/watch/stop', 'POST')['enabled'])
        self.assertFalse(self.server.service.watcher.thread.is_alive())

    def test_private_study_plan_timer_and_judgment_http(self) -> None:
        self.assertEqual(self.request('/api/study')['trials'], [])
        plan = self.request('/api/study/plan', 'POST', {'tasks': ['Find launch decision', 'Find rollback steps']})
        self.assertEqual(len(plan['trials']), 4)
        running = self.request('/api/study/start', 'POST')
        active = next(t for t in running['trials'] if t['state']=='active')
        result = self.request('/api/study/finish', 'POST', {'id':active['id'], 'success':True, 'verified':False, 'corrected':False})
        completed = next(t for t in result['trials'] if t['id']==active['id'])
        self.assertEqual(completed['verified_success'], 0)
        self.assertGreaterEqual(completed['elapsed_seconds'], 0)
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/study/finish', 'POST', {'id':active['id'], 'success':True, 'verified':True, 'corrected':False})
        exported = self.request('/api/study/export')
        self.assertEqual(exported['schema'], 'personal-document-rag-study-aggregate-v1')
        self.assertNotIn('Find launch decision', json.dumps(exported))
        self.request('/api/study/cancel', 'POST')

    def test_decision_history_returns_only_explicit_change_sentences(self) -> None:
        (self.source / 'decision.md').write_text('# Rollout\n2026-04-02: rollout 2026-04-10 → 2026-04-17 changed because approval was delayed.', encoding='utf-8')
        self.request('/api/index', 'POST')
        history = self.request('/api/history?q=rollout')
        event = next(event for event in history['events'] if event['source']['title'] == 'decision')
        self.assertEqual(event['date'], '2026-04-02')
        self.assertEqual(event['before'], 'rollout 2026-04-10')
        self.assertEqual(event['after'], '2026-04-17')
        self.assertEqual(event['reason'], 'approval was delayed')
        self.assertIn('not semantic entity resolution', history['limitations'])

    def test_related_document_bundle_has_no_duplicate_document_paths(self) -> None:
        (self.source / 'spec.md').write_text('# Rollout\nRollback requires approval.', encoding='utf-8')
        (self.source / 'runbook.md').write_text('# Rollout\nRollback steps require approval.', encoding='utf-8')
        self.request('/api/index', 'POST')
        bundle = self.request('/api/bundle?q=rollback%20approval')
        self.assertEqual(len({source['path'] for source in bundle['sources']}), len(bundle['sources']))
        self.assertGreaterEqual(bundle['candidate_documents'], bundle['documents_selected'])
        self.assertIn('no inferred relationships', bundle['provenance'])

    def test_private_review_save_and_human_verdict_http(self) -> None:
        self.request('/api/index', 'POST')
        answer = self.request('/api/ask', 'POST', {'question': 'What does RRF combine?'})
        saved = self.request('/api/reviews/save', 'POST', {'question': 'What does RRF combine?', 'answer': answer, 'experiment': 'candidate'})
        review = saved['reviews'][0]
        self.assertEqual(review['verdict'], 'pending')
        self.assertEqual(review['experiment'], 'candidate')
        self.assertEqual(saved['experiments'][0]['label'], 'candidate')
        self.assertNotIn('context', review['sources'][0])
        judged = self.request('/api/reviews/verdict', 'POST', {'id': review['id'], 'verdict': 'supported', 'notes': 'Checked source'})
        self.assertEqual(judged['summary']['supported'], 1)
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/reviews/verdict', 'POST', {'id': review['id'], 'verdict': 'unsupported'})

    def test_private_review_claim_verdict_http(self) -> None:
        self.request('/api/index', 'POST')
        answer = self.request('/api/ask', 'POST', {'question': 'What does RRF combine?'})
        review = self.request('/api/reviews/save', 'POST', {'question': 'What does RRF combine?', 'answer': answer})['reviews'][0]
        claim = review['claims'][0]
        reviewed = self.request('/api/reviews/claim-verdict', 'POST', {'id': claim['id'], 'verdict': 'supported', 'notes': 'Checked literal quote'})
        self.assertEqual(reviewed['claim_summary']['supported'], 1)

    def test_private_review_export_is_aggregate_only(self) -> None:
        self.request('/api/index', 'POST')
        answer = self.request('/api/ask', 'POST', {'question': 'What does RRF combine?'})
        self.request('/api/reviews/save', 'POST', {'question': 'What does RRF combine?', 'answer': answer})
        exported = self.request('/api/reviews/export')
        self.assertEqual(exported['schema'], 'personal-document-rag-review-aggregate-v1')
        self.assertNotIn('RRF combine', json.dumps(exported))

    def test_unexpected_write_error_has_safe_json_taxonomy(self) -> None:
        with patch.object(self.server.service.reviews, 'save', side_effect=KeyError('private detail')):
            request = urllib.request.Request(self.base + '/api/reviews/save', data=json.dumps({'question': 'Question', 'answer': {'status': 'fallback', 'text': 'Answer', 'sources': []}}).encode(), headers={'Content-Type': 'application/json'}, method='POST')
            with self.assertRaises(urllib.error.HTTPError) as response:
                urllib.request.urlopen(request)
        self.assertEqual(response.exception.code, 500)
        self.assertEqual(json.loads(response.exception.read().decode())['kind'], 'server_error')

    def test_explicit_relations_http_only_accept_active_indexed_documents(self) -> None:
        decision = self.source/'decision.md'
        decision.write_text('# Decision\nLaunch approved.', encoding='utf-8')
        self.request('/api/index', 'POST')
        configured = self.request('/api/relations', 'POST', {'projects': [{'name': 'Aurora', 'decisions': [{'name': 'Launch', 'documents': [str(decision)]}]}]})
        document = configured['projects'][0]['decisions'][0]['documents'][0]
        self.assertEqual(Path(document['path']), decision)
        self.assertEqual(self.request('/api/relations?q=launch')['projects'][0]['name'], 'Aurora')
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/relations', 'POST', {'projects': [{'name': 'Bad', 'decisions': [{'name': 'Bad', 'documents': [str(self.source/'missing.md')]}]}]})

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
        self.assertIn('파일 시스템 수정 시각', page)
        self.assertIn('원문에 적힌 결정 날짜와 다름', page)
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

    def test_background_index_routes_and_invalid_retry(self) -> None:
        started = self.request("/api/index/start", "POST", {"mode": "changed", "strict": True})
        self.assertIn("id", started)
        self.server.service.jobs.wait(3)
        job = self.request("/api/index/job")
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(job["completed"], job["total"])
        self.assertEqual(self.request("/api/status")["documents"], 1)
        self.request("/api/index/start", "POST", {"mode": "full"})
        self.server.service.jobs.wait(3)
        self.assertEqual(self.request("/api/index/job")["summary"]["indexed"], 1)
        with self.assertRaises(urllib.error.HTTPError) as invalid:
            self.request("/api/index/start", "POST", {"mode": "retry"})
        self.assertEqual(invalid.exception.code, 400)
        self.assertEqual(self.request("/api/index/cancel", "POST")["state"], "succeeded")

    def test_original_download_validates_the_returned_snapshot(self) -> None:
        self.request("/api/index", "POST")
        chunk_id = self.request("/api/search?q=RRF")[0]["chunk_id"]
        document = self.source / "guide.md"
        original = document.read_bytes()
        read_bytes = Path.read_bytes

        def changed_before_read(path):
            if path == document:
                path.write_bytes(b"Changed at the download read boundary")
            return read_bytes(path)

        with patch.object(Path, "read_bytes", changed_before_read):
            with self.assertRaises(urllib.error.HTTPError) as changed:
                urllib.request.urlopen(self.base + "/api/file?id=" + chunk_id)
        self.assertEqual(changed.exception.code, 409)

        document.write_bytes(original)
        reads = []
        def changed_after_read(path):
            snapshot = read_bytes(path)
            if path == document:
                reads.append(path)
                path.write_bytes(b"New version after the snapshot was read")
            return snapshot

        with patch.object(Path, "read_bytes", changed_after_read):
            with urllib.request.urlopen(self.base + "/api/file?id=" + chunk_id) as response:
                self.assertEqual(response.read(), original)
        self.assertEqual(len(reads), 1, "Hash validation and response must share one byte snapshot")

    def test_evidence_span_is_escaped_highlighted_and_bounds_checked(self) -> None:
        (self.source / "guide.md").write_text("# Retrieval\nBackground. RRF combines <keyword> and vector rankings.", encoding="utf-8")
        self.request("/api/index", "POST")
        answer = self.request("/api/ask", "POST", {"question": "What does RRF combine?"})
        source = answer["sources"][0]
        route = f"/source?id={source['chunk_id']}&start={source['quote_start']}&end={source['quote_end']}"
        with urllib.request.urlopen(self.base + route) as response:
            page = response.read().decode("utf-8")
        self.assertIn('<mark id="quote">RRF combines &lt;keyword&gt; and vector rankings.</mark>', page)
        with self.assertRaises(urllib.error.HTTPError) as invalid:
            urllib.request.urlopen(self.base + route + "0")
        self.assertEqual(invalid.exception.code, 400)

    def test_pdf_evidence_viewer_links_to_the_indexed_page_fragment(self) -> None:
        write_pdf(self.source / 'report.pdf', 'PDF launch evidence')
        self.request('/api/index', 'POST')
        chunk_id = self.request('/api/search?q=PDF%20launch%20evidence')[0]['chunk_id']
        with urllib.request.urlopen(self.base + '/source?id=' + chunk_id) as response:
            page = response.read().decode('utf-8')
        self.assertIn('page 1', page)
        self.assertIn('/api/file?id=' + chunk_id + '#page=1', page)


if __name__ == "__main__":
    unittest.main()
