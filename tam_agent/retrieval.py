"""Knowledge-base retrieval: a small BM25 index plus metadata-aware ranking.

Why BM25 (keyword search) and not embeddings?
  - Support tickets are full of exact tokens (error codes like E-4102, module names,
    UI terms). Keyword search matches those precisely.
  - It needs no model download or vector DB, builds in well under a second for ~390
    articles, and every score is explainable.
  See SOLUTION.md for how this would change at 10x scale (hybrid BM25 + embeddings).
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:-[0-9]+)?")  # keeps "e-4102" as one token
_STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its
of on or our so that the their them then there these they this to was we were what when where which
who will with you your not no please hi team""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Classic Okapi BM25 over a list of documents."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for term in q:
                if term in tf:
                    f = tf[term]
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
            out.append(s)
        return out


class KnowledgeBase:
    """Wraps the raw articles: id lookup, BM25 search, and metadata boosts."""

    def __init__(self, articles: list[dict]):
        self.articles = articles
        self.by_id = {a["id"]: a for a in articles}
        # Title is repeated so title words weigh more than body words.
        docs = [" ".join([a["meta"].get("title", "")] * 2 + [a["meta"].get("module", ""),
                         " ".join(a["meta"].get("error_codes", []) or []), a["body"]]) for a in articles]
        self.index = BM25(docs)

    def search(self, query: str, module: str = "", error_code: str = "", doc_type: str = "", k: int = 5) -> list[dict]:
        base = self.index.scores(" ".join([query, error_code]))
        ranked = []
        for art, score in zip(self.articles, base):
            meta = art["meta"]
            codes = meta.get("error_codes") or []
            if error_code and error_code.upper() in codes:
                score += 10.0                       # exact error-code match is the strongest signal
            if module and meta.get("module") == module:
                score += 2.0
            if doc_type and meta.get("type") == doc_type:
                score += 2.0
            if meta.get("status") == "deprecated":
                score *= 0.5                        # still findable, but ranked below the current article
            if score > 0:
                ranked.append((score, art))
        ranked.sort(key=lambda x: -x[0])
        return [self.summary(a, round(s, 2)) for s, a in ranked[:k]]

    @staticmethod
    def summary(art: dict, score: float | None = None) -> dict:
        meta = art["meta"]
        out = {"id": art["id"], "title": meta.get("title"), "module": meta.get("module"), "type": meta.get("type"),
               "status": meta.get("status"), "last_updated": meta.get("last_updated"),
               "error_codes": meta.get("error_codes") or [],
               "snippet": re.sub(r"\s+", " ", art["body"].replace("#", ""))[:220]}
        if meta.get("status") == "deprecated":
            out["warning"] = f"DEPRECATED - use {meta.get('superseded_by')} instead"
        if meta.get("status") == "active_incident":
            out["warning"] = "ACTIVE INCIDENT - read this article"
        if score is not None:
            out["score"] = score
        return out