from __future__ import annotations

import json
import math
import os
import urllib.request

from .text import hash_embedding
from .api_requests import APIRequests
from .local_models import LocalEncoder


class Embeddings:
    """One local fallback with an optional OpenAI embedding path; no provider framework."""

    def __init__(self, api_key: str | None = None, model: str | None = None, local_model_path: str | None = None, local_encoder=None):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        if local_model_path and self.api_key:
            raise ValueError("Choose either an OpenAI embedding key or a local embedding model")
        if local_encoder is not None and not local_model_path:
            raise ValueError("A local encoder requires a local model directory")
        self.local = LocalEncoder(local_model_path, local_encoder) if local_model_path else None
        self.mode = f"local:sentence-transformers:{self.local.name}" if self.local else f"openai:{self.model}" if self.api_key else "local:hash-256"
        self.requests = APIRequests()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.local:
            return self.local.embed(texts)
        if not self.api_key:
            return [hash_embedding(text) for text in texts]
        # Conservative UTF-8 byte limits avoid a tokenizer dependency; never truncate evidence.
        sizes = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Embedding inputs must be nonempty strings")
            size = len(text.encode("utf-8"))
            if size > 8000:
                raise ValueError("Embedding input exceeds 8000 UTF-8 bytes. Reduce chunk size or shorten the query.")
            sizes.append(size)
        vectors = []
        start = 0
        while start < len(texts):
            end, batch_bytes = start, 0
            while end < len(texts) and end-start < 64 and batch_bytes+sizes[end] <= 64000:
                batch_bytes += sizes[end]
                end += 1
            batch = self._openai(texts[start:end])
            if vectors and len(batch[0]) != len(vectors[0]):
                raise ValueError("Embedding dimensions changed between batches")
            vectors.extend(batch)
            start = end
        return vectors

    def _openai(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        body = self.requests.send(request)
        items = body.get("data")
        if not isinstance(items, list) or len(items) != len(texts) or any(not isinstance(item, dict) or type(item.get("index")) is not int for item in items):
            raise ValueError("Invalid embedding response items")
        if sorted(item["index"] for item in items) != list(range(len(texts))):
            raise ValueError("Embedding response indexes must cover each input exactly once")
        vectors = [item.get("embedding") for item in sorted(items, key=lambda item: item["index"])]
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise ValueError("Embedding vectors must be nonempty arrays")
        try:
            valid = all(len(vector) == len(vectors[0]) and all(type(value) in {int, float} and math.isfinite(value) for value in vector) for vector in vectors)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Embedding vectors must have consistent dimensions and finite numeric values")
        return vectors
