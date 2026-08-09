"""Tests for the Ollama generation wrapper.

A stub client stands in for the ``ollama`` module, so these run with no daemon,
no model weights and no network.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.generator import Generator  # noqa: E402

NS = 1_000_000_000


class StubClient:
    """Returns a canned Ollama response and records the options it was sent."""

    def __init__(self, payload=None, raises=None):
        self.payload = payload if payload is not None else {
            "response": "  A generated answer.  ",
            "prompt_eval_count": 50,
            "prompt_eval_duration": NS // 2,   # 0.5s -> 100 tok/s
            "eval_count": 20,
            "eval_duration": 2 * NS,           # 2.0s -> 10 tok/s
            "load_duration": 3 * NS,
            "total_duration": 5 * NS,
        }
        self.raises = raises
        self.calls = []

    def generate(self, model, prompt, system, options):
        self.calls.append({"model": model, "prompt": prompt,
                           "system": system, "options": options})
        if self.raises:
            raise self.raises
        return dict(self.payload)


class TestSuccessfulGeneration(unittest.TestCase):
    def setUp(self):
        self.client = StubClient()
        self.gen = Generator(model_name="test-model", client=self.client)
        self.result = self.gen.generate("a prompt")

    def test_response_text_is_stripped(self):
        self.assertEqual(self.result["response"], "A generated answer.")

    def test_not_flagged_as_error(self):
        self.assertFalse(self.result["error"])
        self.assertIsNone(self.result["error_message"])

    def test_token_counts_passed_through(self):
        self.assertEqual(self.result["prompt_tokens"], 50)
        self.assertEqual(self.result["generation_tokens"], 20)

    def test_nanoseconds_converted_to_seconds(self):
        self.assertAlmostEqual(self.result["load_duration"], 3.0)

    def test_throughput_computed_from_ollama_durations(self):
        self.assertAlmostEqual(self.result["prompt_eval_speed"], 100.0)
        self.assertAlmostEqual(self.result["generation_speed"], 10.0)

    def test_load_duration_is_not_folded_into_latency(self):
        # Weight loading is paid once per model, not once per query. Folding it
        # into latency would charge a per-model cost to whichever arm ran first.
        self.assertAlmostEqual(self.result["load_duration"], 3.0)
        self.assertLess(self.result["latency"], 3.0)

    def test_deterministic_options_are_sent(self):
        options = self.client.calls[0]["options"]
        self.assertEqual(options["temperature"], 0.0)
        self.assertEqual(options["seed"], 42)


class TestFailedGeneration(unittest.TestCase):
    def setUp(self):
        self.client = StubClient(raises=ConnectionError("connection refused"))
        self.gen = Generator(model_name="test-model", client=self.client)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.result = self.gen.generate("a prompt")
        self.logged = captured.getvalue()

    def test_error_is_flagged_not_smuggled_into_the_response_text(self):
        # v1 returned "ERROR: <exception>" in the response field, which was
        # then embedded, compared to ground truth and scored as an answer.
        self.assertTrue(self.result["error"])
        self.assertIsNone(self.result["response"])

    def test_error_message_is_preserved_for_diagnosis(self):
        self.assertIn("ConnectionError", self.result["error_message"])

    def test_failure_is_logged(self):
        self.assertIn("Error during generation", self.logged)

    def test_wall_clock_cost_of_a_failure_is_recorded(self):
        # A failed call still consumed time; dropping it would understate the
        # arm's latency in proportion to its failure rate.
        self.assertIsNotNone(self.result["latency"])
        self.assertGreaterEqual(self.result["latency"], 0.0)

    def test_undefined_costs_are_none_not_zero(self):
        # v1 wrote 0.0 here, and those zeros were averaged into throughput
        # means -- pulling them toward zero in proportion to the failure rate,
        # which differs by arm.
        for field in ("prompt_tokens", "generation_tokens",
                      "prompt_eval_speed", "generation_speed"):
            self.assertIsNone(self.result[field], field)


class TestUndefinedThroughput(unittest.TestCase):
    def test_zero_duration_yields_none_not_a_division_error(self):
        client = StubClient({"response": "hi", "prompt_eval_count": 10,
                             "prompt_eval_duration": 0, "eval_count": 5,
                             "eval_duration": 0})
        result = Generator(client=client).generate("p")
        self.assertIsNone(result["prompt_eval_speed"])
        self.assertIsNone(result["generation_speed"])

    def test_missing_fields_yield_none(self):
        result = Generator(client=StubClient({"response": "hi"})).generate("p")
        self.assertIsNone(result["prompt_tokens"])
        self.assertIsNone(result["load_duration"])
        self.assertIsNone(result["generation_speed"])

    def test_empty_response_is_empty_string_not_none(self):
        # An empty generation is a real observation (the evaluator classifies
        # it as an EMPTY failure); it is distinct from a call that errored.
        result = Generator(client=StubClient({"response": ""})).generate("p")
        self.assertEqual(result["response"], "")
        self.assertFalse(result["error"])


class TestWarmup(unittest.TestCase):
    def test_warmup_issues_a_call_and_sets_the_flag(self):
        client = StubClient()
        gen = Generator(client=client)
        self.assertFalse(gen.generate("p")["warmed_up"])
        gen.warmup()
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(gen.generate("p")["warmed_up"])

    def test_warmup_returns_elapsed_seconds(self):
        self.assertGreaterEqual(Generator(client=StubClient()).warmup(), 0.0)

    def test_measured_runs_are_marked_as_warmed(self):
        # The flag lets analysis drop any run that slipped through cold rather
        # than silently averaging a weight-load into a latency mean.
        gen = Generator(client=StubClient())
        gen.warmup()
        self.assertTrue(gen.generate("p")["warmed_up"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
