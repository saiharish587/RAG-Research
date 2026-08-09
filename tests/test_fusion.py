"""Tests for rank and score fusion arithmetic."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.fusion import (  # noqa: E402
    RRF_K,
    min_max_normalise,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)

ident = lambda x: x  # noqa: E731


class TestReciprocalRankFusion(unittest.TestCase):
    """Guards v1 defect: RRF enumerated ranks from 0 instead of 1."""

    def test_rank_is_one_based(self):
        # The top document must score 1/(k+1), not 1/k.
        [(item, score)] = reciprocal_rank_fusion([["a"]], key=ident)
        self.assertEqual(item, "a")
        self.assertAlmostEqual(score, 1.0 / (RRF_K + 1))
        self.assertNotAlmostEqual(score, 1.0 / RRF_K)

    def test_second_rank_scores_less(self):
        results = dict(reciprocal_rank_fusion([["a", "b"]], key=ident))
        self.assertAlmostEqual(results["a"], 1.0 / (RRF_K + 1))
        self.assertAlmostEqual(results["b"], 1.0 / (RRF_K + 2))
        self.assertGreater(results["a"], results["b"])

    def test_agreement_across_lists_accumulates(self):
        results = dict(reciprocal_rank_fusion([["a", "b"], ["a", "c"]], key=ident))
        self.assertAlmostEqual(results["a"], 2.0 / (RRF_K + 1))
        self.assertGreater(results["a"], results["b"])

    def test_consensus_beats_a_single_top_hit(self):
        # "b" is never first but appears in both lists; "a" is first in one.
        fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], key=ident)
        self.assertEqual(fused[0][0], "b")

    def test_ordering_is_descending(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"]], key=ident)
        scores = [s for _, s in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_n_truncates(self):
        fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], key=ident, top_n=2)
        self.assertEqual([i for i, _ in fused], ["a", "b"])

    def test_deduplicates_by_key(self):
        fused = reciprocal_rank_fusion([["a", "a"]], key=ident)
        self.assertEqual(len(fused), 1)

    def test_works_with_dict_items_and_a_key_function(self):
        lists = [
            [{"chunk": {"text": "x"}}, {"chunk": {"text": "y"}}],
            [{"chunk": {"text": "y"}}],
        ]
        fused = reciprocal_rank_fusion(lists, key=lambda r: r["chunk"]["text"])
        self.assertEqual(fused[0][0]["chunk"]["text"], "y")

    def test_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([], key=ident), [])
        self.assertEqual(reciprocal_rank_fusion([[]], key=ident), [])

    def test_tie_break_is_deterministic(self):
        # Same score for both; order must not depend on dict iteration luck.
        first = reciprocal_rank_fusion([["a"], ["b"]], key=ident)
        second = reciprocal_rank_fusion([["a"], ["b"]], key=ident)
        self.assertEqual([i for i, _ in first], [i for i, _ in second])


class TestMinMaxNormalise(unittest.TestCase):
    def test_scales_to_unit_range(self):
        self.assertEqual(min_max_normalise([2.0, 4.0, 6.0]), [0.0, 0.5, 1.0])

    def test_constant_input_does_not_divide_by_zero(self):
        self.assertEqual(min_max_normalise([3.0, 3.0]), [0.0, 0.0])

    def test_empty(self):
        self.assertEqual(min_max_normalise([]), [])

    def test_negative_values(self):
        self.assertEqual(min_max_normalise([-1.0, 0.0, 1.0]), [0.0, 0.5, 1.0])


class TestWeightedScoreFusion(unittest.TestCase):
    """Guards v1 defect: dense and sparse scores had incomparable ranges."""

    # Score ranges measured on the v1 run: FAISS cosine occupied roughly
    # [0.55, 0.90] while TF-IDF cosine occupied roughly [0.00, 0.20].
    DENSE = [0.90, 0.78, 0.72, 0.55]
    SPARSE = [0.02, 0.18, 0.05, 0.00]

    @staticmethod
    def _effective_sparse_share(dense, sparse, dense_weight, normalise):
        """Fraction of the fused score's dynamic range owned by the sparse arm.

        Fusion is linear, so each arm can be recovered through the public API
        by weighting it at 1.0 and the other at 0.0. The share each arm
        actually contributes is then its weight times its own spread.
        """
        dense_arm = weighted_score_fusion(dense, sparse, dense_weight=1.0, normalise=normalise)
        sparse_arm = weighted_score_fusion(dense, sparse, dense_weight=0.0, normalise=normalise)
        dense_span = dense_weight * (max(dense_arm) - min(dense_arm))
        sparse_span = (1.0 - dense_weight) * (max(sparse_arm) - min(sparse_arm))
        return sparse_span / (dense_span + sparse_span)

    def test_unnormalised_fusion_silently_shrinks_the_sparse_weight(self):
        # The v1 defect stated precisely: `0.7*dense + 0.3*sparse` on these
        # ranges gives the sparse arm a nominal 30% of the decision but an
        # effective share near 18%, because its spread is half the dense
        # spread. The blend was not the blend that was reported.
        share = self._effective_sparse_share(self.DENSE, self.SPARSE, 0.7, normalise=False)
        self.assertAlmostEqual(share, 0.054 / 0.299, places=6)
        self.assertLess(share, 0.20)

    def test_normalisation_restores_the_declared_weight(self):
        # After min-max scaling both arms span [0, 1], so the effective share
        # equals the declared weight exactly.
        share = self._effective_sparse_share(self.DENSE, self.SPARSE, 0.7, normalise=True)
        self.assertAlmostEqual(share, 0.30, places=9)

    def test_declared_weight_is_honoured_across_settings(self):
        for dense_weight in (0.2, 0.5, 0.7, 0.9):
            share = self._effective_sparse_share(
                self.DENSE, self.SPARSE, dense_weight, normalise=True
            )
            self.assertAlmostEqual(share, 1.0 - dense_weight, places=9,
                                   msg=f"dense_weight={dense_weight}")

    def test_sparse_decides_when_dense_is_indifferent(self):
        # Dense cannot separate these two chunks; the lexical signal should
        # therefore determine the order. Unnormalised it cannot, because a
        # near-zero dense spread still swamps a near-zero sparse spread only
        # after scaling makes the comparison meaningful.
        dense = [0.8000, 0.8001]
        sparse = [0.01, 0.19]
        fused = weighted_score_fusion(dense, sparse, dense_weight=0.7)
        self.assertGreater(fused[1], fused[0])

    def test_weights_sum_to_one(self):
        fused = weighted_score_fusion([1.0, 0.0], [1.0, 0.0], dense_weight=0.7)
        self.assertAlmostEqual(fused[0], 1.0)
        self.assertAlmostEqual(fused[1], 0.0)

    def test_pure_dense_weighting(self):
        fused = weighted_score_fusion([1.0, 0.0], [0.0, 1.0], dense_weight=1.0)
        self.assertEqual(fused, [1.0, 0.0])

    def test_pure_sparse_weighting(self):
        fused = weighted_score_fusion([1.0, 0.0], [0.0, 1.0], dense_weight=0.0)
        self.assertEqual(fused, [0.0, 1.0])

    def test_normalisation_can_be_disabled(self):
        fused = weighted_score_fusion([0.8, 0.2], [0.1, 0.1], dense_weight=0.5, normalise=False)
        self.assertAlmostEqual(fused[0], 0.5 * 0.8 + 0.5 * 0.1)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            weighted_score_fusion([1.0], [1.0, 2.0])

    def test_invalid_weight_raises(self):
        with self.assertRaises(ValueError):
            weighted_score_fusion([1.0], [1.0], dense_weight=1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
