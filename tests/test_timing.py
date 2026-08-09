"""Tests for stage-level run timing.

A fake clock is injected throughout, so these assert exact arithmetic rather
than tolerances against real elapsed time -- no sleeps, no flakiness.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.timing import StageTimer  # noqa: E402


class FakeClock:
    """Monotonic clock advanced explicitly by the test."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestStageTimer(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.timer = StageTimer(clock=self.clock)

    def test_single_stage_records_its_duration(self):
        with self.timer.stage("generate"):
            self.clock.advance(2.5)
        self.assertAlmostEqual(self.timer.stages["generate"], 2.5)
        self.assertEqual(self.timer.counts["generate"], 1)

    def test_repeated_stage_accumulates_and_counts(self):
        for seconds in (1.0, 2.0, 3.0):
            with self.timer.stage("retrieve"):
                self.clock.advance(seconds)
        self.assertAlmostEqual(self.timer.stages["retrieve"], 6.0)
        self.assertEqual(self.timer.counts["retrieve"], 3)

    def test_sibling_stages_sum_to_accounted(self):
        with self.timer.stage("rewrite"):
            self.clock.advance(0.5)
        with self.timer.stage("retrieve"):
            self.clock.advance(0.25)
        with self.timer.stage("generate"):
            self.clock.advance(4.0)
        self.assertAlmostEqual(self.timer.accounted, 4.75)

    def test_elapsed_covers_untimed_gaps(self):
        self.clock.advance(1.0)  # work outside any stage
        with self.timer.stage("generate"):
            self.clock.advance(2.0)
        self.clock.advance(0.5)  # more work outside any stage
        self.assertAlmostEqual(self.timer.elapsed, 3.5)
        self.assertAlmostEqual(self.timer.accounted, 2.0)
        self.assertAlmostEqual(self.timer.unaccounted(), 1.5)

    def test_nesting_is_rejected(self):
        # A nested stage would be counted inside its parent as well as on its
        # own, inflating the total. This must fail loudly, not silently.
        with self.assertRaises(RuntimeError):
            with self.timer.stage("outer"):
                with self.timer.stage("inner"):
                    pass

    def test_timer_is_reusable_after_a_failed_stage(self):
        with self.assertRaises(ValueError):
            with self.timer.stage("generate"):
                self.clock.advance(1.0)
                raise ValueError("model exploded")
        # The stage must still be recorded: a failed generation costs real time
        # and dropping it would understate the arm's latency.
        self.assertAlmostEqual(self.timer.stages["generate"], 1.0)
        # And the timer must not be stuck inside the aborted stage.
        with self.timer.stage("recover"):
            self.clock.advance(0.5)
        self.assertAlmostEqual(self.timer.stages["recover"], 0.5)

    def test_report_is_flat_and_csv_friendly(self):
        with self.timer.stage("rewrite"):
            self.clock.advance(0.5)
        with self.timer.stage("generate"):
            self.clock.advance(2.0)
        report = self.timer.report()
        self.assertAlmostEqual(report["stage_rewrite_s"], 0.5)
        self.assertEqual(report["stage_rewrite_calls"], 1)
        self.assertAlmostEqual(report["stage_generate_s"], 2.0)
        self.assertAlmostEqual(report["pipeline_latency_s"], 2.5)
        self.assertTrue(all(isinstance(v, (int, float)) for v in report.values()))

    def test_report_exposes_unaccounted_time(self):
        self.clock.advance(3.0)  # entirely untimed work
        report = self.timer.report()
        self.assertAlmostEqual(report["unaccounted_latency_s"], 3.0)

    def test_no_stages_is_not_an_error(self):
        self.clock.advance(1.0)
        self.assertEqual(self.timer.stages, {})
        self.assertAlmostEqual(self.timer.report()["pipeline_latency_s"], 1.0)

    def test_default_clock_is_monotonic(self):
        # time.time() can step backwards under NTP adjustment, which would
        # yield negative durations.
        import time as time_module

        timer = StageTimer()
        self.assertIs(timer._clock, time_module.perf_counter)


class TestTimingAnswersTheV1Defect(unittest.TestCase):
    """The v1 number was the final generate() call, not the run."""

    def test_multi_call_pipeline_totals_all_llm_calls(self):
        clock = FakeClock()
        timer = StageTimer(clock=clock)
        # An advanced-arm run: rewrite, retrieve, rerank, then answer.
        with timer.stage("rewrite"):
            clock.advance(1.2)
        with timer.stage("retrieve"):
            clock.advance(0.3)
        with timer.stage("rerank"):
            clock.advance(0.8)
        with timer.stage("generate"):
            clock.advance(2.0)

        # v1 would have reported 2.0 -- the final call alone.
        self.assertAlmostEqual(timer.stages["generate"], 2.0)
        # The run actually cost 4.3s, and the difference is not noise.
        self.assertAlmostEqual(timer.elapsed, 4.3)
        self.assertGreater(timer.elapsed, 2 * timer.stages["generate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
