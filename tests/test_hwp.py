from __future__ import annotations

import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.documents import parse_document
from rag.hwp import HWPTextConverter
from rag.service import RAGService


class HWPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.executable = self.root / 'hwp5txt.exe'
        self.executable.write_bytes(b'fixture')
        self.document = self.root / 'decision.hwp'
        self.document.write_bytes(b'HWP fixture')

    def tearDown(self):
        self.temporary.cleanup()

    def converter(self):
        def runner(command, **options):
            self.assertEqual(command[1], '--output')
            self.assertEqual(Path(command[-1]), self.document)
            Path(command[2]).write_text('HWP launch decision', encoding='utf-8')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        return HWPTextConverter(str(self.executable), runner)

    def test_binary_hwp_uses_explicit_converter_and_preserves_noninferred_location(self):
        parsed = parse_document(self.document, hwp=self.converter())
        self.assertEqual([(section.location, section.text) for section in parsed.sections], [('HWP text', 'HWP launch decision')])
        with self.assertRaisesRegex(ValueError, 'requires --hwp-executable'):
            parse_document(self.document)

    def test_conversion_errors_and_background_index_propagation_are_visible(self):
        failed = HWPTextConverter(str(self.executable), lambda command, **options: subprocess.CompletedProcess(command, 1, stdout='', stderr='bad HWP'))
        with self.assertRaisesRegex(ValueError, 'exit code 1'):
            failed.text(self.document)
        service = RAGService(self.root, self.root/'index.db', mode='offline')
        converter = self.converter()
        service.index.hwp = converter
        try:
            with patch('rag.index.parse_document', wraps=parse_document) as parser:
                self.assertEqual(service.index_documents().indexed, 1)
            self.assertIs(parser.call_args.kwargs['hwp'], converter)
            self.assertIn('hwp:hwp5txt:hwp5txt.exe', service.index.processing_version)
        finally:
            service.close()
