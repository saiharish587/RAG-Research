"""Tests for the Evaluator class.

The embedding model is injected as a stub, so these run with no GPU, no model
weights and no sentence-transformers installation.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluator import Evaluator  # noqa: E402


class StubEncoder:
    """Deterministic bag-of-characters encoder standing in for BGE.

    Real embeddings are not needed: the evaluator's contract is *when* cosine
    is computed and how failures are handled, not what the encoder returns.
    """

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [[float(t.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]


class ExplodingEncoder:
    def encode(self, texts, **kwargs):
        raise RuntimeError("CUDA out of memory")


CONTEXT = ["Docker isolates processes using kernel namespaces and cgroups."]
GROUND_TRUTH = "Docker isolates processes using namespaces."


class TestEvaluatorSuccessPath(unittest.TestCase):
    def setUp(self):
        self.encoder = StubEncoder()
        self.ev = Evaluator(embedding_model=self.encoder)

    def test_all_metrics_present_for_good_answer(self):
        row = self.ev.evaluate("Docker isolates processes using namespaces.", GROUND_TRUTH, CONTEXT)
        self.assertEqual(row["status"], "ok")
        self.assertFalse(row["is_failure"])
        for field in ("token_f1", "exact_match", "cosine", "groundedness",
                      "context_precision", "context_recall", "retrieval_hit_rate"):
            self.assertIsNotNone(row[field], field)

    def test_perfect_answer_scores_f1_one(self):
        row = self.ev.evaluate(GROUND_TRUTH, GROUND_TRUTH, CONTEXT)
        self.assertAlmostEqual(row["token_f1"], 1.0)
        self.assertEqual(row["exact_match"], 1.0)

    def test_cost_counters_are_passed_through(self):
        row = self.ev.evaluate("An answer.", GROUND_TRUTH, CONTEXT,
                               generation_tokens=42, prompt_tokens=1085, latency=1.5)
        self.assertEqual(row["generation_tokens"], 42)
        self.assertEqual(row["prompt_tokens"], 1085)
        self.assertEqual(row["latency"], 1.5)

    def test_refusal_is_flagged(self):
        row = self.ev.evaluate("I don't know.", GROUND_TRUTH, CONTEXT)
        self.assertTrue(row["is_refusal"])
        self.assertEqual(row["status"], "ok")  # a refusal is a real generation


class TestEvaluatorFailurePath(unittest.TestCase):
    """Guards v1 defect #1: failures scored 0.0 and averaged into accuracy."""

    def setUp(self):
        self.ev = Evaluator(embedding_model=StubEncoder())

    def test_empty_response_yields_none_not_zero(self):
        row = self.ev.evaluate("", GROUND_TRUTH, CONTEXT)
        self.assertEqual(row["status"], "empty")
        self.assertTrue(row["is_failure"])
        for field in ("token_f1", "exact_match", "cosine", "groundedness"):
            self.assertIsNone(row[field], f"{field} must be None, not 0.0")

    def test_whitespace_response_yields_none(self):
        row = self.ev.evaluate("  \n\n  ", GROUND_TRUTH, CONTEXT)
        self.assertEqual(row["status"], "empty")
        self.assertIsNone(row["token_f1"])

    def test_error_response_yields_none(self):
        row = self.ev.evaluate(None, GROUND_TRUTH, CONTEXT, error=True)
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["token_f1"])

    def test_failure_still_records_cost(self):
        # The 44-second whitespace loops really did cost 44 seconds.
        row = self.ev.evaluate("   ", GROUND_TRUTH, CONTEXT,
                               generation_tokens=3657, latency=44.4)
        self.assertEqual(row["latency"], 44.4)
        self.assertEqual(row["generation_tokens"], 3657)

    def test_encoder_not_called_for_failed_generation(self):
        encoder = StubEncoder()
        Evaluator(embedding_model=encoder).evaluate("", GROUND_TRUTH, CONTEXT)
        self.assertEqual(encoder.calls, [])


class TestNoRagArm(unittest.TestCase):
    """Guards against structural zeros on the arm with no retrieval."""

    def setUp(self):
        self.ev = Evaluator(embedding_model=StubEncoder())

    def test_context_metrics_are_none_without_retrieval(self):
        row = self.ev.evaluate("Docker isolates processes.", GROUND_TRUTH, None)
        self.assertEqual(row["status"], "ok")
        for field in ("groundedness", "context_precision", "context_recall", "retrieval_hit_rate"):
            self.assertIsNone(row[field], f"{field} must be None on the no-RAG arm")

    def test_quality_metrics_still_computed_without_retrieval(self):
        row = self.ev.evaluate("Docker isolates processes.", GROUND_TRUTH, None)
        self.assertIsNotNone(row["token_f1"])
        self.assertIsNotNone(row["cosine"])


class TestEncoderRobustness(unittest.TestCase):
    def test_encoder_failure_gives_none_not_zero(self):
        ev = Evaluator(embedding_model=ExplodingEncoder())
        # The evaluator logs the failure; silence it so a deliberately
        # triggered error does not look like a real one in test output.
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            row = ev.evaluate("An answer.", GROUND_TRUTH, CONTEXT)
        self.assertIn("CUDA out of memory", captured.getvalue(),
                      "the failure must still be logged, not swallowed silently")
        self.assertIsNone(row["cosine"], "an infrastructure error must not look like a wrong answer")
        self.assertIsNotNone(row["token_f1"], "other metrics must survive an encoder failure")

    def test_cosine_disabled_when_no_model_available(self):
        ev = Evaluator(embedding_model=None)
        # Touching .model would try to import sentence_transformers; instead
        # confirm lazy construction has not happened at __init__ time.
        self.assertIsNone(ev._model)


class TestEvaluateRunCompatibility(unittest.TestCase):
    """The driver calls evaluate_run(...); its signature must not drift."""

    def setUp(self):
        self.ev = Evaluator(embedding_model=StubEncoder())

    def test_returns_raw_fields_and_metrics(self):
        row = self.ev.evaluate_run(
            "What does Docker do?",
            "Docker isolates processes.",
            CONTEXT,
            GROUND_TRUTH,
            {"latency": 1.2, "generation_tokens": 10, "prompt_tokens": 900,
             "generation_speed": 8.3, "prompt_eval_speed": 120.0},
        )
        self.assertEqual(row["query"], "What does Docker do?")
        self.assertEqual(row["ground_truth"], GROUND_TRUTH)
        self.assertEqual(row["n_retrieved"], 1)
        self.assertEqual(row["latency"], 1.2)
        self.assertEqual(row["generation_speed"], 8.3)
        self.assertIsNotNone(row["token_f1"])

    def test_missing_stats_do_not_crash(self):
        row = self.ev.evaluate_run("q", "an answer", None, GROUND_TRUTH)
        self.assertEqual(row["n_retrieved"], 0)
        self.assertIsNone(row["latency"])

    def test_v1_metric_names_are_gone(self):
        # hallucination_rate was 1 - accuracy; precision_at_k was not
        # rank-aware. Both names must not reappear.
        row = self.ev.evaluate_run("q", "an answer", CONTEXT, GROUND_TRUTH, {})
        for stale in ("hallucination_rate", "precision_at_k", "recall_at_k", "answer_accuracy"):
            self.assertNotIn(stale, row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
