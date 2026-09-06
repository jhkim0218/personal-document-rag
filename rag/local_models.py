from __future__ import annotations

import math
from pathlib import Path


def _model_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Local model directory does not exist: {directory}")
    return directory


def _vectors(values, expected: int) -> list[list[float]]:
    try:
        vectors = [list(vector) for vector in values]
    except TypeError as error:
        raise ValueError("Local embedding model returned invalid vectors") from error
    if len(vectors) != expected or not vectors or any(not vector for vector in vectors):
        raise ValueError("Local embedding model returned an unexpected vector count")
    try:
        valid = all(len(vector) == len(vectors[0]) and all(type(value) in {int, float} and math.isfinite(value) for value in vector) for vector in vectors)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Local embedding vectors must have consistent finite numeric values")
    return vectors


class LocalEncoder:
    """A local-files-only sentence-transformers adapter, loaded only when selected."""

    def __init__(self, path: str | Path, encoder=None):
        self.path = _model_directory(path)
        self.name = self.path.name
        self.encoder = encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError("Local embeddings require 'sentence-transformers'. Install requirements-local.txt; no model will be downloaded automatically.") from error
            self.encoder = SentenceTransformer(str(self.path), local_files_only=True)
        return _vectors(self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False), len(texts))


class LocalReranker:
    """A local-files-only cross-encoder adapter for final candidate ordering."""

    def __init__(self, path: str | Path, encoder=None):
        self.path = _model_directory(path)
        self.name = self.path.name
        self.encoder = encoder

    def scores(self, query: str, texts: list[str]) -> list[float]:
        if self.encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError("Local reranking requires 'sentence-transformers'. Install requirements-local.txt; no model will be downloaded automatically.") from error
            self.encoder = CrossEncoder(str(self.path), local_files_only=True)
        values = self.encoder.predict([(query, text) for text in texts], show_progress_bar=False)
        try:
            scores = [float(value) for value in values]
        except (TypeError, ValueError) as error:
            raise ValueError("Local reranker returned invalid scores") from error
        if len(scores) != len(texts) or any(not math.isfinite(value) for value in scores):
            raise ValueError("Local reranker returned invalid scores")
        return scores
