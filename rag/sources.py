from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .documents import SUPPORTED_EXTENSIONS


DEFAULT_EXCLUDES = (".git", ".local", ".venv", "venv", "__pycache__", ".cache", "node_modules")


@dataclass(frozen=True)
class Source:
    path: str
    enabled: bool = True
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()


class SourceSettings:
    def __init__(self, sources: list[Source], extensions: tuple[str, ...]):
        self.sources = sources
        self.extensions = extensions

    @classmethod
    def from_dict(cls, data: dict, *, require_existing: bool = True) -> "SourceSettings":
        if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
            raise ValueError("sources must be a list")
        extensions = data.get("extensions", sorted(SUPPORTED_EXTENSIONS))
        if not isinstance(extensions, list) or not extensions or any(not isinstance(e, str) or e.lower() not in SUPPORTED_EXTENSIONS for e in extensions):
            raise ValueError("Choose supported document extensions")
        sources = []
        seen = set()
        for item in data["sources"]:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].strip():
                raise ValueError("Each source needs a folder path")
            root = Path(item["path"]).expanduser().resolve()
            enabled = item.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError("enabled must be a boolean")
            if require_existing and enabled and not root.is_dir():
                raise ValueError(f"Source folder does not exist: {root}")
            if str(root) in seen:
                raise ValueError(f"Duplicate source folder: {root}")
            seen.add(str(root))
            filters = {}
            for key in ("includes", "excludes"):
                values = item.get(key, [])
                if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                    raise ValueError(f"{key} must contain relative subfolder paths")
                normalized = []
                for value in values:
                    relative = Path(value)
                    if relative.is_absolute() or relative.drive or ".." in relative.parts:
                        raise ValueError(f"{key} cannot escape its source folder")
                    normalized.append(relative.as_posix())
                filters[key] = tuple(normalized)
            sources.append(Source(str(root), enabled, **filters))
        return cls(sources, tuple(sorted(set(e.lower() for e in extensions))))

    def as_dict(self) -> dict:
        return {"sources": [{**asdict(source), "includes": list(source.includes), "excludes": list(source.excludes)} for source in self.sources], "extensions": list(self.extensions)}

    def allows(self, path: str | Path, source: Source | None = None) -> bool:
        path = Path(path).resolve()
        if path.suffix.lower() not in self.extensions or path.name.startswith(("~$", ".~")) or path.name.endswith("~"):
            return False
        for candidate in [source] if source else self.sources:
            root = Path(candidate.path)
            if not candidate.enabled or not path.is_relative_to(root):
                continue
            relative = path.relative_to(root)
            if any(part.lower() in DEFAULT_EXCLUDES for part in relative.parts):
                continue
            if any(relative.is_relative_to(Path(excluded)) for excluded in candidate.excludes):
                continue
            if candidate.includes and not any(relative.is_relative_to(Path(included)) for included in candidate.includes):
                continue
            return True
        return False

    def files(self, source: Source) -> list[Path]:
        if not source.enabled:
            return []
        root = Path(source.path)
        if not root.is_dir():
            raise ValueError(f"Source folder is unavailable: {root}")
        files = []
        # Do not follow directory symlinks or silently prune an unreadable tree as if it were empty.
        def failed(error):
            raise error
        for folder, directories, names in os.walk(root, followlinks=False, onerror=failed):
            directories[:] = [name for name in directories if name.lower() not in DEFAULT_EXCLUDES and not (Path(folder) / name).is_symlink()]
            files.extend(Path(folder) / name for name in names if self.allows(Path(folder) / name, source))
        return sorted(files)

    def preview(self) -> dict:
        files = sorted({str(path.resolve()) for source in self.sources for path in self.files(source)})
        return {"files": files, "count": len(files), "default_excludes": list(DEFAULT_EXCLUDES)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
