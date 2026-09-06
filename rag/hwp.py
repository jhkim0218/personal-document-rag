from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class HWPTextConverter:
    """Explicit local HWP v5-to-text conversion; no bundled or downloaded converter."""

    def __init__(self, executable: str, runner=subprocess.run):
        candidate = Path(executable).expanduser()
        resolved = str(candidate.resolve()) if candidate.is_file() else shutil.which(executable)
        if not resolved:
            raise ValueError(f'HWP converter executable was not found: {executable}')
        self.executable = resolved
        self.runner = runner
        self.mode = f'hwp5txt:{Path(resolved).name}'

    def text(self, path: Path) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'document.txt'
            try:
                completed = self.runner([self.executable, '--output', str(output), str(path)], capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired as error:
                raise ValueError('HWP conversion timed out') from error
            if completed.returncode:
                raise ValueError(f'HWP conversion failed with exit code {completed.returncode}')
            if not output.is_file():
                raise ValueError('HWP converter did not create a text output file')
            text = output.read_text(encoding='utf-8', errors='replace').strip()
        if not text:
            raise ValueError('HWP converter produced no text')
        return text
