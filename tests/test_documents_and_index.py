from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from rag.documents import parse_document, to_chunks
from rag.index import RAGIndex
from rag.text import hash_embedding


class TestEmbeddings:
    def __init__(self, mode: str):
        self.mode = mode

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embedding(text) for text in texts]


def write_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = f'<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


def write_pdf(path: Path, text: str) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(('BT /F1 12 Tf 72 720 Td (' + text + ') Tj ET').encode('latin-1'))} >>\nstream\nBT /F1 12 Tf 72 720 Td ({text}) Tj ET\nendstream".encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode())
        content.extend(obj)
        content.extend(b"\nendobj\n")
    start_xref = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    content.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    content.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start_xref}\n%%EOF".encode())
    path.write_bytes(content)


class DocumentAndIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "notes.md").write_text("# Retrieval\nreciprocal rank fusion combines BM25 and vector search.", encoding="utf-8")
        (self.root / "plan.txt").write_text("The launch decision is Friday.", encoding="utf-8")
        write_docx(self.root / "policy.docx", ["Document retention is thirty days."])
        write_pdf(self.root / "report.pdf", "PDF extraction works")
        self.index = RAGIndex(self.root / "index.sqlite3")

    def tearDown(self) -> None:
        self.index.close()
        self.temporary.cleanup()

    def test_parses_all_document_types_with_source_locations(self) -> None:
        for filename, location_prefix in (("notes.md", "heading:"), ("plan.txt", "text"), ("policy.docx", "paragraph"), ("report.pdf", "page")):
            document = parse_document(self.root / filename)
            chunks = to_chunks(document)
            self.assertTrue(chunks)
            self.assertTrue(chunks[0][0].startswith(location_prefix), chunks[0][0])

    def test_incremental_indexing_reindexes_changes_and_removes_deleted_documents(self) -> None:
        first = self.index.index_directory(self.root)
        self.assertEqual(first.indexed, 4)
        self.assertEqual(self.index.status()["documents"], 4)
        second = self.index.index_directory(self.root)
        self.assertEqual(second.indexed, 0)
        self.assertEqual(second.skipped, 4)
        (self.root / "plan.txt").write_text("The launch decision changed to Monday.", encoding="utf-8")
        (self.root / "policy.docx").unlink()
        third = self.index.index_directory(self.root)
        self.assertEqual(third.indexed, 1)
        self.assertEqual(third.removed, 1)
        self.assertEqual(self.index.status()["documents"], 3)

    def test_hybrid_search_returns_evidence_and_source_metadata(self) -> None:
        self.index.index_directory(self.root)
        result = self.index.search("What does reciprocal rank fusion combine?", mode="hybrid")[0]
        self.assertEqual(Path(result.path).name, "notes.md")
        self.assertGreater(result.overlap, 0)
        source = self.index.source(result.chunk_id)
        self.assertEqual(source["location"], "heading: Retrieval")

    def test_same_content_files_receive_distinct_documents(self) -> None:
        (self.root / "copy.txt").write_text("The launch decision is Friday.", encoding="utf-8")
        self.index.index_directory(self.root)
        self.assertEqual(self.index.status()["documents"], 5)

    def test_embedding_mode_change_reindexes_unchanged_documents(self) -> None:
        self.index.close()
        database = self.root / "mode.sqlite3"
        first_index = RAGIndex(database, embeddings=TestEmbeddings("test:first"))
        self.assertEqual(first_index.index_directory(self.root).indexed, 4)
        first_index.close()
        second_index = RAGIndex(database, embeddings=TestEmbeddings("test:second"))
        self.assertEqual(second_index.index_directory(self.root).indexed, 4)
        self.assertEqual(second_index.status()["embedding_mode"], "test:second")
        second_index.close()
        self.index = RAGIndex(self.root / "index.sqlite3")


if __name__ == "__main__":
    unittest.main()
