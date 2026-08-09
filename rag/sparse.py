"""Sparse (lexical) retrieval index used by the advanced pipeline.

Kept in its own module for two reasons.

**Fit-once semantics are a property of the index, not of the search call.** In
v1 ``TfidfVectorizer().fit_transform(corpus)`` ran inside the per-query search
path, so the whole corpus was re-vectorised on every query. Because that work
sat outside the timed region it never appeared in the reported latency either.
Here the vectoriser is fit exactly once, when the index object is constructed,
and the pipeline caches the object against the chunk set it was built from.

**scikit-learn is an optional dependency at test time.** The pipeline accepts
any object matching this module's small protocol -- ``__len__`` and
``scores(query) -> list[float]`` -- so the caching and fusion logic can be
exercised with a lightweight stand-in on machines without scikit-learn
installed. The real benchmark always uses :class:`TfidfIndex`.
"""

from __future__ import annotations

from typing import Iterable, Sequence


class TfidfIndex:
    """TF-IDF cosine index over a fixed corpus, fit once at construction."""

    def __init__(self, texts: Iterable[str], stop_words: str | None = "english"):
        # Imported here, not at module scope, so that importing this module --
        # and therefore the pipeline -- does not require scikit-learn.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.texts: list[str] = list(texts)
        self._vectorizer = TfidfVectorizer(stop_words=stop_words)
        # L2-normalised by default in scikit-learn, so a dot product against a
        # transformed query is already a cosine similarity.
        self._matrix = self._vectorizer.fit_transform(self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def scores(self, query: str) -> list[float]:
        """Cosine similarity of ``query`` against every indexed text."""
        query_vector = self._vectorizer.transform([query])
        return (self._matrix * query_vector.T).toarray().ravel().tolist()


def corpus_signature(texts: Sequence[str]) -> tuple:
    """Identify a corpus so a cached index can be invalidated when it changes.

    Length alone is not sufficient: a rebuilt index with the same number of
    chunks but different content would otherwise be scored against a stale
    vocabulary. Python caches string hashes, so for a stable chunk list this is
    a cheap pointer-level operation after the first call.
    """
    return (len(texts), hash(tuple(texts)))
