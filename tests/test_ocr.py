from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import documents
from rag.ocr import TesseractOCR
from rag.service import RAGService


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.executable = self.root / 'tesseract.exe'
        self.executable.write_bytes(b'fixture')

    def tearDown(self):
        self.temporary.cleanup()

    def test_image_only_pdf_page_uses_local_tesseract_and_retains_page_location(self):
        commands = []

        def runner(command, **options):
            commands.append((command, options))
            self.assertTrue(Path(command[1]).is_file())
            return subprocess.CompletedProcess(command, 0, stdout='OCR launch decision', stderr='')

        page = types.SimpleNamespace(extract_text=lambda: '', images=[types.SimpleNamespace(name='scan.png', data=b'png')])
        reader = types.SimpleNamespace(pages=[page])
        pypdf = types.ModuleType('pypdf')
        pypdf.PdfReader = lambda path: reader
        errors = types.ModuleType('pypdf.errors')
        errors.PyPdfError = ValueError
        with patch.dict(sys.modules, {'pypdf': pypdf, 'pypdf.errors': errors}):
            sections = documents._parse_pdf(self.root/'scan.pdf', TesseractOCR(str(self.executable), 'eng', runner))
        self.assertEqual([(section.location, section.text) for section in sections], [('page 1 · OCR', 'OCR launch decision')])
        self.assertEqual(commands[0][0][-2:], ['-l', 'eng'])

    def test_ocr_errors_are_page_specific_and_not_silently_indexed(self):
        page = types.SimpleNamespace(images=[types.SimpleNamespace(name='scan.png', data=b'png')])
        engine = TesseractOCR(str(self.executable), runner=lambda command, **options: subprocess.CompletedProcess(command, 2, stdout='', stderr='bad image'))
        with self.assertRaisesRegex(ValueError, 'OCR page 3 failed'):
            engine.page_text(page, 3)
        with self.assertRaisesRegex(ValueError, 'executable was not found'):
            TesseractOCR(str(self.root/'missing.exe'))

    def test_background_index_receives_the_selected_ocr_pipeline(self):
        (self.root/'guide.md').write_text('# Guide\nOCR configuration test.', encoding='utf-8')
        service = RAGService(self.root, self.root/'index.db', mode='offline')
        engine = types.SimpleNamespace(mode='fixture-ocr')
        service.index.ocr = engine
        try:
            with patch('rag.index.parse_document', wraps=documents.parse_document) as parser:
                self.assertEqual(service.index_documents().indexed, 1)
            self.assertIs(parser.call_args.kwargs['ocr'], engine)
            self.assertIn('ocr:fixture-ocr', service.index.processing_version)
        finally:
            service.close()
