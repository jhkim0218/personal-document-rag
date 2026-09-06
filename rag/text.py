from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+|[가-힣]+")
STOPWORDS = {"의", "은", "는", "이", "가", "을", "를", "에", "와", "과", "도", "로", "한", "할", "때", "및", "무엇인가", "알려줘"}


def tokenize(text: str) -> list[str]:
    """Return stable tokens for Korean and Latin text without external NLP data."""
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS and not (len(token) == 1 and "가" <= token <= "힣")]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 160) -> list[str]:
    """Keep paragraph boundaries where possible, with a small overlap for context."""
    if type(max_chars) is not int or type(overlap_chars) is not int or max_chars < 1 or not 0 <= overlap_chars < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap_chars < max_chars")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text.strip()]:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current += "\n\n" + paragraph
        else:
            chunks.extend(_split_long_text(current, max_chars, overlap_chars))
            current = paragraph
    if current:
        chunks.extend(_split_long_text(current, max_chars, overlap_chars))
    return chunks


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def hash_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Deterministic offline fallback. Configure an API embedding provider for semantic quality."""
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    features: Iterable[tuple[str, float]] = [(token, 1.0) for token in tokens]
    characters = re.sub(r"\s+", "", text.lower())
    features = list(features) + [(characters[index : index + 3], 0.25) for index in range(max(0, len(characters) - 2))]
    for feature, weight in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[bucket] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def token_overlap(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def keyword_score(query: str, text: str) -> float:
    query_counts = Counter(tokenize(query))
    text_counts = Counter(tokenize(text))
    return sum(min(count, text_counts[token]) for token, count in query_counts.items()) / max(1, sum(query_counts.values()))


def bm25_scores(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Small in-process BM25 for a personal corpus; avoids a service dependency."""
    return BM25(texts).scores(query, k1, b)


class BM25:
    def __init__(self, texts: list[str], tokenizer=tokenize):
        self.tokenizer = tokenizer
        self.documents = [Counter(tokenizer(text)) for text in texts]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.documents)) or 1.0
        self.frequency = Counter(term for document in self.documents for term in document)

    def scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        terms = self.tokenizer(query)
        scores = []
        for document, length in zip(self.documents, self.lengths):
            score = 0.0
            for term in terms:
                frequency = document[term]
                if frequency:
                    inverse = math.log(1 + (len(self.documents) - self.frequency[term] + 0.5) / (self.frequency[term] + 0.5))
                    score += inverse * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length / self.average_length))
            scores.append(score)
        return scores


def retrieval_tokens(text: str) -> list[str]:
    """Supplementary lexical features; does not change stored hash embeddings."""
    tokens = tokenize(text)
    # Small explicit terminology dictionary, not a claim of learned semantics.
    compact = re.sub(r"[\s_-]+", "", text.lower())
    for canonical, aliases in {"rrf": ("rrf", "reciprocalrankfusion"), "rto": ("rto", "recoverytimeobjective", "복구시간목표")}.items():
        if any(alias in tokens if len(alias) <= 3 else alias in compact for alias in aliases):
            tokens.extend(["syn:" + canonical] * 3)
    for korean in re.findall(r"[가-힣]+", text):
        if len(korean) == 1 and korean not in STOPWORDS:
            tokens.append(korean)
    # Whitespace-insensitive Korean bigrams supplement exact words, never split error codes.
    for run in re.findall(r"[가-힣]+(?:\s+[가-힣]+)*", text):
        joined = re.sub(r"\s+", "", run)
        tokens.extend("ko:" + joined[i:i+2] for i in range(len(joined)-1))
    return tokens
