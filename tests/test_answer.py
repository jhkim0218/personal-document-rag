from __future__ import annotations

import unittest
from unittest.mock import patch
import json

from rag.answer import Answerer, validate_citations
from rag.embeddings import Embeddings
from rag.index import SearchResult


def result(text: str, overlap: float = 1.0, keyword_score: float = 1.0) -> SearchResult:
    return SearchResult("chunk-1", "C:/docs/decision.md", "decision", "heading: Decision", text, 1.0, keyword_score, 0.7, overlap)


class AnswerTests(unittest.TestCase):
    def test_extractive_fallback_has_traceable_citations(self) -> None:
        answer = Answerer(api_key="").answer("When is launch?", [result("The launch is Friday. The owner is Mina.")])
        self.assertEqual(answer.status, "fallback")
        self.assertIn("[1]", answer.text)
        self.assertEqual(answer.sources[0]["location"], "heading: Decision")

    def test_missing_evidence_abstains(self) -> None:
        answer = Answerer(api_key="").answer("What is annual revenue?", [result("The launch is Friday.", overlap=0.0, keyword_score=0.0)])
        self.assertEqual(answer.status, "abstained")
        self.assertFalse(answer.sources)

    def test_invalid_citation_is_rejected(self) -> None:
        self.assertIsNone(validate_citations("Supported statement [1]", 1))
        self.assertEqual(validate_citations("Unsupported [2]", 1), "The generated answer included an invalid citation number")
        self.assertEqual(validate_citations("No source", 1), "The generated answer did not include any citations")

    def test_competing_date_claims_are_flagged_for_source_review(self) -> None:
        first = result("The launch date is 2026-10-01.")
        second = SearchResult("chunk-2", "C:/docs/revised-decision.md", "revised", "heading: Decision", "The launch date is 2026-10-08.", 0.9, 1.0, 0.7, 1.0)
        answer = Answerer(api_key="").answer("What is the launch date?", [first, second])
        self.assertEqual(answer.status, "conflict")
        self.assertIn("서로 다른 날짜", answer.text)

    def test_openai_response_request_is_grounded_and_not_stored(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                return json.dumps({"output": [{"type": "message", "content": [{"type": "output_text", "text": "The launch is Friday. [1]"}]}]}).encode()

        with patch("rag.answer.urllib.request.urlopen", return_value=Response()) as mocked:
            answer = Answerer(api_key="test-key", model="test-model").answer("When is launch?", [result("The launch is Friday.")])
        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["model"], "test-model")
        self.assertFalse(payload["store"])
        self.assertEqual(answer.status, "answered")

    def test_openai_embedding_request_preserves_input_order(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                return json.dumps({"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}).encode()

        with patch("rag.embeddings.urllib.request.urlopen", return_value=Response()) as mocked:
            vectors = Embeddings(api_key="test-key", model="embedding-test").embed(["first", "second"])
        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(payload["input"], ["first", "second"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
