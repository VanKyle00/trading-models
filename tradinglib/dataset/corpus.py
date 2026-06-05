"""RAG corpus: discover project docs, chunk them, build a TF-IDF cosine index.

TF-IDF (sklearn, already a dep) keeps the base install lean - no torch. The
``search(query, k)`` interface is stable; a dense embedder can replace the
vectorizer later (sub-project 4) without changing callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tradinglib.data.paths import repo_root

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def discover_docs() -> list[Path]:
    root = repo_root()
    paths: list[Path] = []
    for rel in ("README.md", "MODELS.md"):
        p = root / rel
        if p.exists():
            paths.append(p)
    paths += sorted((root / "docs").glob("*.md"))
    paths += sorted((root / "docs" / "models").glob("*.html"))
    paths += sorted((root / "docs" / "concepts").glob("*.html"))
    paths += sorted(root.glob("models/*/*/model.md"))
    return paths


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    clean = _WS.sub(" ", _TAG.sub(" ", text)).strip()
    if not clean:
        return []
    return [clean[i : i + max_chars] for i in range(0, len(clean), max_chars)]


@dataclass
class Corpus:
    chunks: list[str]
    _vectorizer: TfidfVectorizer
    _matrix: object

    @classmethod
    def from_chunks(cls, chunks: list[str]) -> Corpus:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(chunks)
        return cls(chunks=chunks, _vectorizer=vec, _matrix=matrix)

    @classmethod
    def build(cls) -> Corpus:
        chunks: list[str] = []
        for path in discover_docs():
            chunks += chunk_text(path.read_text(encoding="utf-8", errors="ignore"))
        return cls.from_chunks(chunks)

    def search(self, query: str, k: int = 3) -> list[str]:
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        ranked = sorted(range(len(self.chunks)), key=lambda i: sims[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if sims[i] > 0]
