"""Tests for the aggregation layer.

The central guarantee under test: a generation failure rate can never be
laundered into a quality score. This is the defect that produced the v1
"RAG Loop Stabilization Effect".
"""

import math
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.aggregate import (  # noqa: E402
    aggregate_cost,
    aggregate_quality,
    failure_breakdown,
    summarise,
)


def row(query, status="ok", f1=None, latency=1.0, **extra):
    """Build a result row shaped like Evaluator output."""
    failed = status != "ok"
    base = {
        "query": query,
        "status": status,
        "is_failure": failed,
        "is_refusal": False,
        "token_f1": None if failed else f1,
        "exact_match": None if failed else 0.0,
        "cosine": None if failed else 0.85,
        "groundedness": None if failed else 0.5,
        "context_precision": None if failed else 0.04,
        "context_recall": None if failed else 0.3,
        "retrieval_hit_rate": None if failed else 1.0,
        "latency": latency,
        "generation_tokens": 100,
        "prompt_tokens": 1000,
    }
    base.update(extra)
    return base


class TestFailureBreakdown(unittest.TestCase):
    def test_counts_and_rate(self):
        rows = [row("q1"), row("q2", status="empty"), row("q3", status="empty")]
        out = failure_breakdown(rows)
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["n_failed"], 2)
        self.assertAlmostEqual(out["failure_rate"], 2 / 3)

    def test_reports_denominator(self):
        # A rate without its denominator is not reportable.
        self.assertEqual(failure_breakdown([row("q1")])["n"], 1)

    def test_status_split_is_preserved(self):
        rows = [row("q1"), row("q2", status="empty"), row("q3", status="error")]
        self.assertEqual(failure_breakdown(rows)["by_status"],
                         {"ok": 1, "empty": 1, "error": 1})

    def test_refusals_counted_separately(self):
        rows = [row("q1", is_refusal=True), row("q2")]
        out = failure_breakdown(rows)
        self.assertEqual(out["n_refusal"], 1)
        self.assertEqual(out["n_failed"], 0, "a refusal is not a generation failure")

    def test_empty_input(self):
        self.assertEqual(failure_breakdown([])["n"], 0)


class TestQualityExcludesFailures(unittest.TestCase):
    """The core regression guard for the v1 headline defect."""

    def test_failures_do_not_drag_down_the_mean(self):
        # 1 good answer at F1=0.9, plus 9 empty generations.
        rows = [row("q1", f1=0.9)] + [row(f"q{i}", status="empty") for i in range(2, 11)]
        quality = aggregate_quality(rows)
        self.assertAlmostEqual(quality["token_f1"]["mean"], 0.9)
        self.assertEqual(quality["token_f1"]["n"], 1)

    def test_v1_behaviour_would_have_reported_0_09(self):
        # Documents the magnitude of the old error: treating failures as zeros
        # turns a 0.90 quality score into 0.09.
        rows = [row("q1", f1=0.9)] + [row(f"q{i}", status="empty") for i in range(2, 11)]
        v1_style = sum(r["token_f1"] or 0.0 for r in rows) / len(rows)
        v2 = aggregate_quality(rows)["token_f1"]["mean"]
        self.assertAlmostEqual(v1_style, 0.09)
        self.assertAlmostEqual(v2, 0.9)
        self.assertGreater(v2 - v1_style, 0.8)

    def test_quality_and_failure_rate_reported_together(self):
        rows = [row("q1", f1=0.9)] + [row(f"q{i}", status="empty") for i in range(2, 11)]
        out = summarise([{**r, "model": "M", "rag_type": "no_rag"} for r in rows])
        entry = out[("M", "no_rag")]
        self.assertAlmostEqual(entry["quality"]["token_f1"]["mean"], 0.9)
        self.assertAlmostEqual(entry["failures"]["failure_rate"], 0.9)

    def test_all_failures_gives_none_not_zero(self):
        rows = [row(f"q{i}", status="empty") for i in range(5)]
        quality = aggregate_quality(rows)
        self.assertIsNone(quality["token_f1"]["mean"])
        self.assertEqual(quality["token_f1"]["n"], 0)


class TestUnitOfAnalysis(unittest.TestCase):
    """Guards v1 defect #5: deterministic repeats counted as independent."""

    def test_repeats_of_one_query_count_once(self):
        rows = [row("same-query", f1=0.5) for _ in range(30)]
        self.assertEqual(aggregate_quality(rows, group_by_query=True)["token_f1"]["n"], 1)

    def test_per_run_unit_counts_every_row(self):
        rows = [row("same-query", f1=0.5) for _ in range(30)]
        self.assertEqual(aggregate_quality(rows, group_by_query=False)["token_f1"]["n"], 30)

    def test_query_unit_averages_within_query_first(self):
        rows = [row("q1", f1=0.0), row("q1", f1=1.0), row("q2", f1=0.5)]
        quality = aggregate_quality(rows, group_by_query=True)
        self.assertEqual(quality["token_f1"]["n"], 2)
        self.assertAlmostEqual(quality["token_f1"]["mean"], 0.5)

    def test_unit_is_labelled(self):
        rows = [row("q1", f1=0.5)]
        self.assertEqual(aggregate_quality(rows)["token_f1"]["unit"], "query")
        self.assertEqual(aggregate_quality(rows, group_by_query=False)["token_f1"]["unit"], "run")

    def test_pseudo_replication_shrinks_the_interval(self):
        # Demonstrates why the unit matters. This mirrors the real v1 layout:
        # 13 queries x 30 deterministic repeats. Because temperature=0 with a
        # fixed seed makes the repeats byte-identical, each query contributes
        # one value repeated 30 times.
        rows = [
            row(f"q{q}", f1=0.20 + 0.03 * q, latency=1.0)
            for q in range(13)
            for _ in range(30)
        ]
        by_query = aggregate_quality(rows, group_by_query=True)["token_f1"]
        by_run = aggregate_quality(rows, group_by_query=False)["token_f1"]

        self.assertEqual(by_query["n"], 13)
        self.assertEqual(by_run["n"], 390)
        # Identical data, identical mean -- only the claimed precision differs.
        self.assertAlmostEqual(by_query["mean"], by_run["mean"])

        width_q = by_query["ci_high"] - by_query["ci_low"]
        width_r = by_run["ci_high"] - by_run["ci_low"]
        self.assertGreater(width_q, width_r)

        # The narrowing is exactly sqrt((Q*R - 1) / (Q - 1)) for Q queries each
        # repeated R times. Derivation: with x_q constant across repeats and
        # SS = sum_q (x_q - mean)^2, the per-query half-width is proportional to
        # sqrt(SS/(Q-1))/sqrt(Q) and the per-run half-width to
        # sqrt(R*SS/(QR-1))/sqrt(QR); the ratio simplifies to the above.
        # It tends to sqrt(R) for large Q -- the Bessel correction is why it is
        # 5.694 here rather than sqrt(30) = 5.477.
        q, r = 13, 30
        self.assertAlmostEqual(width_q / width_r, math.sqrt((q * r - 1) / (q - 1)), places=6)


class TestCostAggregation(unittest.TestCase):
    """Guards v1 defect #3: a single outlier drove the headline latency claim."""

    def test_median_reported_alongside_mean(self):
        rows = [row(f"q{i}", f1=0.5, latency=1.0) for i in range(10)]
        stats = aggregate_cost(rows)["latency"]
        self.assertIn("median", stats)
        self.assertAlmostEqual(stats["median"], 1.0)

    def test_single_outlier_moves_mean_not_median(self):
        rows = [row(f"q{i}", f1=0.5, latency=1.0) for i in range(9)]
        rows.append(row("q9", f1=0.5, latency=226.8))
        stats = aggregate_cost(rows)["latency"]
        self.assertGreater(stats["mean"], 20)
        self.assertAlmostEqual(stats["median"], 1.0)

    def test_confidence_interval_exposes_the_instability(self):
        rows = [row(f"q{i}", f1=0.5, latency=1.0) for i in range(9)]
        rows.append(row("q9", f1=0.5, latency=226.8))
        stats = aggregate_cost(rows)["latency"]
        self.assertLess(stats["ci_low"], 0, "interval must reveal that the mean is not resolvable")

    def test_failures_included_in_cost_by_default(self):
        rows = [row("q1", f1=0.5, latency=1.0), row("q2", status="empty", latency=44.4)]
        self.assertEqual(aggregate_cost(rows)["latency"]["n"], 2)
        self.assertEqual(aggregate_cost(rows, include_failures=False)["latency"]["n"], 1)

    def test_cost_unit_is_run(self):
        rows = [row("q1", f1=0.5, latency=1.0)]
        self.assertEqual(aggregate_cost(rows)["latency"]["unit"], "run")


class TestSummarise(unittest.TestCase):
    def test_groups_by_model_and_rag_type(self):
        rows = [
            {**row("q1", f1=0.4), "model": "A", "rag_type": "naive"},
            {**row("q2", f1=0.6), "model": "A", "rag_type": "naive"},
            {**row("q1", f1=0.2), "model": "B", "rag_type": "no_rag"},
        ]
        out = summarise(rows)
        self.assertEqual(set(out), {("A", "naive"), ("B", "no_rag")})
        self.assertAlmostEqual(out[("A", "naive")]["quality"]["token_f1"]["mean"], 0.5)

    def test_every_group_has_failures_quality_and_cost(self):
        rows = [{**row("q1", f1=0.4), "model": "A", "rag_type": "naive"}]
        entry = summarise(rows)[("A", "naive")]
        self.assertEqual(set(entry), {"failures", "quality", "cost"})

    def test_all_aggregates_carry_dispersion(self):
        # No bare means anywhere: n, sd and an interval must always be present.
        rows = [{**row(f"q{i}", f1=0.4 + i * 0.05), "model": "A", "rag_type": "naive"}
                for i in range(5)]
        entry = summarise(rows)[("A", "naive")]
        stats = entry["quality"]["token_f1"]
        for key in ("n", "mean", "sd", "ci_low", "ci_high"):
            self.assertIn(key, stats)
            self.assertIsNotNone(stats[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
