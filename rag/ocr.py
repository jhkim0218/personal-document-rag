from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class TesseractOCR:
    """Explicit local Tesseract OCR for image-only PDF pages; never a network service."""

    def __init__(self, executable: str = 'tesseract', language: str = 'eng', runner=subprocess.run):
        candidate = Path(executable).expanduser()
        resolved = str(candidate.resolve()) if candidate.is_file() else shutil.which(executable)
        if not resolved:
            raise ValueError(f"OCR executable was not found: {executable}")
        if not re.fullmatch(r'[A-Za-z0-9_+.-]{1,40}', language):
            raise ValueError('OCR language must be a Tesseract language code')
        self.executable = resolved
        self.language = language
        self.runner = runner
        self.mode = f'tesseract:{language}'

    def page_text(self, page, number: int) -> str:
        images = list(getattr(page, 'images', ()))
        if not images:
            raise ValueError(f'OCR page {number} has no embedded image')
        texts = []
        for image in images:
            suffix = Path(getattr(image, 'name', '')).suffix or '.png'
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / f'image{suffix}'
                target.write_bytes(image.data)
                try:
                    completed = self.runner([self.executable, str(target), 'stdout', '-l', self.language], capture_output=True, text=True, timeout=60)
                except subprocess.TimeoutExpired as error:
                    raise ValueError(f'OCR page {number} timed out') from error
            if completed.returncode:
                raise ValueError(f'OCR page {number} failed with Tesseract exit code {completed.returncode}')
            if completed.stdout.strip():
                texts.append(completed.stdout.strip())
        if not texts:
            raise ValueError(f'OCR page {number} produced no text')
        return '\n'.join(texts)
