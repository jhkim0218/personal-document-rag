from __future__ import annotations

import json
import os
import urllib.request

from .text import hash_embedding


class Embeddings:
    """One local fallback with an optional OpenAI embedding path; no provider framework."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self.mode = f"openai:{self.model}" if self.api_key else "local:hash-256"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._openai(texts) if self.api_key else [hash_embedding(text) for text in texts]

    def _openai(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
        vectors = [item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])]
        if len(vectors) != len(texts):
            raise ValueError("Embedding response count did not match input count")
        return vectors
