"""Tests for the sparse index module.

:class:`rag.sparse.TfidfIndex` needs scikit-learn, which is not present in
every environment, so those tests skip when it is missing. The corpus
signature is pure Python and is always tested -- it is the part that decides
whether a cached index is stale, so a defect there would silently score
queries against the wrong vocabulary.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.sparse import TfidfIndex, corpus_signature  # noqa: E402

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class TestCorpusSignature(unittest.TestCase):
    def test_identical_corpora_match(self):
        a = ["one", "two", "three"]
        b = ["one", "two", "three"]
        self.assertEqual(corpus_signature(a), corpus_signature(b))

    def test_appending_a_chunk_changes_the_signature(self):
        base = ["one", "two"]
        self.assertNotEqual(corpus_signature(base), corpus_signature(base + ["three"]))

    def test_editing_a_chunk_changes_the_signature(self):
        # The case a length-only cache key misses.
        self.assertNotEqual(
            corpus_signature(["one", "two"]),
            corpus_signature(["one", "different"]),
        )

    def test_reordering_changes_the_signature(self):
        # Order is meaningful: it is the row index used to map scores back to
        # chunks, so a reordered corpus must invalidate the cache.
        self.assertNotEqual(
            corpus_signature(["one", "two"]),
            corpus_signature(["two", "one"]),
        )

    def test_empty_corpus(self):
        self.assertEqual(corpus_signature([]), corpus_signature([]))


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class TestTfidfIndex(unittest.TestCase):
    CORPUS = [
        "Docker containers package an application with its dependencies.",
        "Kubernetes orchestrates containers across a cluster of machines.",
        "Continuous integration runs the test suite on every commit.",
    ]

    def setUp(self):
        self.index = TfidfIndex(self.CORPUS)

    def test_length_matches_corpus(self):
        self.assertEqual(len(self.index), 3)

    def test_one_score_per_document(self):
        self.assertEqual(len(self.index.scores("containers")), 3)

    def test_lexical_match_scores_highest(self):
        scores = self.index.scores("Kubernetes cluster orchestration")
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 1)

    def test_unrelated_query_scores_zero(self):
        self.assertTrue(all(s == 0.0 for s in self.index.scores("zebra oscilloscope")))

    def test_scores_are_cosines_in_unit_range(self):
        for query in ("containers", "test suite", "docker application"):
            for score in self.index.scores(query):
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0 + 1e-9)

    def test_repeated_queries_are_stable(self):
        self.assertEqual(self.index.scores("containers"), self.index.scores("containers"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
