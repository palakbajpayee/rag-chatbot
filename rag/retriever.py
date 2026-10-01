"""Pure-Python BM25 retriever (no numpy, no vector database)."""
import math
import re
from collections import Counter

from rag.chunker import Chunk

_STOPWORDS = frozenset("""
a an and are as at be but by can do does for from has have how i if in into is it its
me my of on or our so than that the their them then there these they this to us was
we what when where which who why will with you your yours am been being did i'm i've
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Tiny suffix-stripper so 'refunds', 'refunded', 'refunding' all match 'refund'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index the heading together with the body so section titles boost relevance.
        self._docs = [Counter(tokenize(f"{c.section} {c.section} {c.text}")) for c in chunks]
        self._lens = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scored = []
        for chunk, doc, length in zip(self.chunks, self._docs, self._lens):
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                    score += self._idf[term] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
