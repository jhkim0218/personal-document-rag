from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from .text import chunk_text


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx"}


@dataclass(frozen=True)
class ParsedSection:
    location: str
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    path: Path
    title: str
    content_hash: str
    sections: list[ParsedSection]


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_document(path: Path) -> ParsedDocument:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    if extension == ".pdf":
        sections = _parse_pdf(path)
    elif extension == ".docx":
        sections = _parse_docx(path)
    elif extension in {".md", ".markdown"}:
        sections = _parse_markdown(path)
    else:
        sections = [ParsedSection("text", path.read_text(encoding="utf-8", errors="replace"))]
    sections = [section for section in sections if section.text.strip()]
    if not sections:
        raise ValueError("No extractable text found")
    return ParsedDocument(path=path, title=path.stem, content_hash=content_hash(path), sections=sections)


def to_chunks(document: ParsedDocument, max_chars: int = 900, overlap_chars: int = 160) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for section in document.sections:
        for number, text in enumerate(chunk_text(section.text, max_chars, overlap_chars), start=1):
            suffix = f" · chunk {number}" if len(section.text) > max_chars else ""
            chunks.append((f"{section.location}{suffix}", text))
    return chunks


def _parse_markdown(path: Path) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    current_heading = "document"
    buffer: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            if buffer:
                sections.append(ParsedSection(current_heading, "\n".join(buffer)))
            current_heading = f"heading: {match.group(2).strip()}"
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append(ParsedSection(current_heading, "\n".join(buffer)))
    return sections


def _parse_docx(path: Path) -> list[ParsedSection]:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    sections: list[ParsedSection] = []
    for number, paragraph in enumerate(root.iter(f"{namespace}p"), start=1):
        text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t")).strip()
        if text:
            sections.append(ParsedSection(f"paragraph {number}", text))
    return sections


def _parse_pdf(path: Path) -> list[ParsedSection]:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - exercised in manual setup
        raise RuntimeError("PDF support requires `pip install -r requirements.txt`") from error
    reader = PdfReader(str(path))
    return [ParsedSection(f"page {number}", page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
