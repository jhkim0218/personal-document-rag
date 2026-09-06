from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from .text import chunk_text


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx", ".pptx", ".xlsx", ".hwp", ".hwpx"}


@dataclass(frozen=True)
class Chunking:
    strategy: str = "structured"
    max_chars: int = 900
    overlap_chars: int = 160

    def __post_init__(self):
        if self.strategy not in {"structured", "fixed"}:
            raise ValueError("Chunk strategy must be structured or fixed")
        if type(self.max_chars) is not int or type(self.overlap_chars) is not int or not 1 <= self.max_chars <= 16000 or not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("Require 1 <= max_chars <= 16000 and 0 <= overlap_chars < max_chars")


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


def parse_document(path: Path, ocr=None, hwp=None) -> ParsedDocument:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    if extension == ".pdf":
        sections = _parse_pdf(path, ocr=ocr)
    elif extension == ".docx":
        sections = _parse_docx(path)
    elif extension == ".pptx":
        sections = _parse_pptx(path)
    elif extension == ".xlsx":
        sections = _parse_xlsx(path)
    elif extension == ".hwpx":
        sections = _parse_hwpx(path)
    elif extension == ".hwp":
        if not hwp:
            raise ValueError('Binary HWP requires --hwp-executable or conversion to HWPX/TXT')
        sections = [ParsedSection('HWP text', hwp.text(path))]
    elif extension in {".md", ".markdown"}:
        sections = _parse_markdown(path)
    else:
        sections = [ParsedSection("text", path.read_text(encoding="utf-8", errors="replace"))]
    sections = [section for section in sections if section.text.strip()]
    if not sections:
        raise ValueError("No extractable text found")
    return ParsedDocument(path=path, title=path.stem, content_hash=content_hash(path), sections=sections)


def to_chunks(document: ParsedDocument, max_chars: int = 900, overlap_chars: int = 160, strategy: str = "structured") -> list[tuple[str, str]]:
    Chunking(strategy, max_chars, overlap_chars)
    chunks: list[tuple[str, str]] = []
    if strategy == "fixed":
        combined = "\n\n".join(section.text for section in document.sections)
        spans = []
        cursor = 0
        for section in document.sections:
            spans.append((cursor, cursor + len(section.text), section.location))
            cursor += len(section.text) + 2
        for start in range(0, len(combined), max_chars-overlap_chars):
            end = min(start+max_chars, len(combined))
            text = combined[start:end]
            if text.strip():
                locations = list(dict.fromkeys(location for left, right, location in spans if left < end and right > start))
                chunks.append((" → ".join(locations) + f" · chars {start}:{end}", text))
            if end == len(combined):
                break
        return chunks
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


def _parse_pptx(path: Path) -> list[ParsedSection]:
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            ((int(match.group(1)), name) for name in archive.namelist()
             if (match := re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name))),
            key=lambda item: item[0],
        )
        sections = []
        for number, name in slides:
            root = ElementTree.fromstring(archive.read(name))
            text = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t")).strip()
            if text:
                sections.append(ParsedSection(f"slide {number}", text))
    return sections


def _parse_xlsx(path: Path) -> list[ParsedSection]:
    sheet_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    package_rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter() if node.tag.endswith("}t"))
                      for item in shared_root.iter(f"{sheet_ns}si")]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships.iter(f"{package_rel_ns}Relationship")}
        sections = []
        for sheet in workbook.iter(f"{sheet_ns}sheet"):
            target = targets.get(sheet.attrib.get(f"{rel_ns}id", ""))
            if not target:
                raise ValueError("XLSX worksheet relationship is missing")
            root = ElementTree.fromstring(archive.read("xl/" + target.lstrip("/")))
            for cell in root.iter(f"{sheet_ns}c"):
                reference = cell.attrib.get("r")
                if not reference:
                    continue
                value = cell.findtext(f"{sheet_ns}v")
                if cell.attrib.get("t") == "s" and value is not None:
                    try:
                        value = shared[int(value)]
                    except (ValueError, IndexError) as error:
                        raise ValueError(f"Invalid XLSX shared string at cell {reference}") from error
                elif cell.attrib.get("t") == "inlineStr":
                    inline = cell.find(f"{sheet_ns}is")
                    value = "".join(node.text or "" for node in inline.iter() if node.tag.endswith("}t")) if inline is not None else None
                formula = cell.findtext(f"{sheet_ns}f")
                text = (f"={formula} → {value}" if formula and value is not None else f"={formula}" if formula else value or "").strip()
                if text:
                    sections.append(ParsedSection(f"sheet: {sheet.attrib.get('name', 'unnamed')} · cell {reference}", text))
    return sections


def _parse_hwpx(path: Path) -> list[ParsedSection]:
    with zipfile.ZipFile(path) as archive:
        sections = []
        for number, name in sorted(
            ((int(match.group(1)), name) for name in archive.namelist()
             if (match := re.fullmatch(r"Contents/section(\d+)\.xml", name))),
            key=lambda item: item[0],
        ):
            root = ElementTree.fromstring(archive.read(name))
            text = "".join(node.text or "" for node in root.iter() if node.tag.endswith("}t")).strip()
            if text:
                sections.append(ParsedSection(f"section {number + 1}", text))
    return sections


def _parse_pdf(path: Path, ocr=None) -> list[ParsedSection]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as error:  # pragma: no cover - exercised in manual setup
        raise RuntimeError("PDF support requires `pip install -r requirements.txt`") from error
    try:
        reader = PdfReader(str(path))
        sections = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append(ParsedSection(f"page {number}", text))
            elif ocr:
                sections.append(ParsedSection(f"page {number} · OCR", ocr.page_text(page, number)))
            else:
                sections.append(ParsedSection(f"page {number}", text))
        return sections
    except PyPdfError as error:
        raise ValueError(f"PDF parsing failed: {error}") from error
