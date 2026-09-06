from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from rag.embeddings import Embeddings
from rag.index import RAGIndex
from rag.text import BM25, tokenize


class SearchQualityTests(unittest.TestCase):
    def test_bm25_matches_original_equation(self):
        texts = ["alpha alpha beta", "beta gamma", "alpha", ""]
        documents = [Counter(tokenize(text)) for text in texts]
        average = sum(sum(d.values()) for d in documents) / len(documents)
        for query in ("alpha", "alpha beta", "alpha alpha", "missing", ""):
            expected = []
            for document in documents:
                score = 0
                for term in tokenize(query):
                    frequency = document[term]
                    df = sum(term in d for d in documents)
                    if frequency:
                        score += math.log(1 + (len(documents)-df+0.5)/(df+0.5)) * frequency * 2.5 / (frequency+1.5*(0.25+0.75*sum(document.values())/average))
                expected.append(score)
            for actual, reference in zip(BM25(texts).scores(query), expected):
                self.assertAlmostEqual(actual, reference)

    def test_cache_scores_and_invalidation_across_writer_scope_and_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / "a", root / "b"
            a.mkdir(); b.mkdir()
            (a / "a.txt").write_text("alpha launch Friday", encoding="utf-8")
            (b / "b.txt").write_text("beta launch Monday", encoding="utf-8")
            index = RAGIndex(root / "index.db", Embeddings(api_key=""))
            writer = RAGIndex(root / "index.db", Embeddings(api_key=""))
            try:
                writer.index_directory(a); writer.index_directory(b)
                for mode in ("keyword", "vector", "hybrid"):
                    for lexical in ("legacy", "enhanced"):
                        cached = index.search("launch", mode=mode, lexical=lexical)
                        uncached = index.search("launch", mode=mode, lexical=lexical, use_cache=False)
                        self.assertEqual(cached, uncached)
                before = index.cache_builds
                with patch.object(index.embeddings, "embed", side_effect=AssertionError("Keyword embedding call")), patch("rag.index.json.loads", side_effect=AssertionError("Vector JSON decoded on warm keyword search")):
                    index.search("alpha", mode="keyword")
                    index.search("beta", mode="keyword")
                self.assertEqual(index.cache_builds, before)
                (a / "a.txt").write_text("alpha launch Tuesday", encoding="utf-8")
                writer.index_directory(a)
                self.assertIn("Tuesday", index.search("alpha", mode="keyword")[0].text)
                self.assertGreater(index.cache_builds, before)
                index.path_filter = lambda path: Path(path).parent == b
                self.assertEqual([Path(r.path).name for r in index.search("launch", mode="keyword")], ["b.txt"])
                (b / "b.txt").unlink()
                writer.index_directory(b)
                self.assertEqual(index.search("launch", mode="keyword"), [])
                self.assertIsNone(index.connection.execute("SELECT name FROM sqlite_master WHERE name='chunks_fts'").fetchone())
            finally:
                writer.close(); index.close()

    def test_korean_synonym_filename_and_exact_code_development_cases(self):
        fixture = json.loads(Path("data/eval/retrieval_cases.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, text in fixture["documents"].items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            index = RAGIndex(root / "index.db", Embeddings(api_key=""))
            try:
                index.index_directory(root)
                for case in fixture["queries"]:
                    with self.subTest(case=case["id"]):
                        results = index.search(case["query"], mode="keyword", rerank=False)
                        self.assertTrue(results)
                        self.assertEqual(Path(results[0].path).relative_to(root).as_posix(), case["expected"])
                        if case["type"] == "identifier":
                            hybrid = index.search(case["query"], mode="hybrid", rerank=True)
                            self.assertEqual(Path(hybrid[0].path).relative_to(root).as_posix(), case["expected"])
            finally:
                index.close()


if __name__ == "__main__":
    unittest.main()
