from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.preflight import check_runtime


class PreflightTests(unittest.TestCase):
    def test_optional_features_are_incomplete_without_existing_runtimes(self):
        with patch('scripts.preflight.existing_executable', return_value=None), patch('scripts.preflight.importlib.util.find_spec', return_value=None):
            report = check_runtime({'local-model', 'ocr', 'hwp'}, embedding_model='missing')
        self.assertFalse(report['ready'])
        self.assertFalse(report['checks']['local-model']['ready'])
        self.assertFalse(report['checks']['ocr']['ready'])
        self.assertFalse(report['checks']['hwp']['ready'])

    def test_existing_paths_and_required_runtimes_make_the_check_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)/'model'
            model.mkdir()
            with patch('scripts.preflight.existing_executable', return_value='C:/tools/tool.exe'), patch('scripts.preflight.importlib.util.find_spec', return_value=object()):
                report = check_runtime({'local-model', 'ocr', 'hwp'}, embedding_model=str(model))
        self.assertTrue(report['ready'])
