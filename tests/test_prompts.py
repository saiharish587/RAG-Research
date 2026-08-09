"""Tests for prompt construction and parity across pipeline arms.

Prompt parity is an experimental control, so it is asserted mechanically rather
than maintained by convention.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.prompts import (  # noqa: E402
    ABSTENTION_INSTRUCTION,
    SYSTEM_PERSONA,
    build_answer_prompt,
    build_routing_prompt,
    build_subquery_prompt,
    format_context,
)

QUERY = "What is a container?"
CONTEXT = ["Containers package an application with its dependencies.", "They share the host kernel."]


class TestPromptParity(unittest.TestCase):
    """Guards v1 defect #8: no_rag used a bare prompt, RAG arms did not."""

    def test_no_rag_prompt_has_the_persona(self):
        prompt = build_answer_prompt(QUERY, None)
        self.assertIn(SYSTEM_PERSONA, prompt)

    def test_no_rag_prompt_has_the_abstention_instruction(self):
        prompt = build_answer_prompt(QUERY, None)
        self.assertIn(ABSTENTION_INSTRUCTION, prompt)

    def test_rag_prompt_has_both_too(self):
        prompt = build_answer_prompt(QUERY, CONTEXT)
        self.assertIn(SYSTEM_PERSONA, prompt)
        self.assertIn(ABSTENTION_INSTRUCTION, prompt)

    def test_only_difference_is_the_context_block(self):
        # The core parity assertion: strip the context block from the RAG
        # prompt and only the task verb should differ.
        with_ctx = build_answer_prompt(QUERY, CONTEXT)
        without_ctx = build_answer_prompt(QUERY, None)
        self.assertIn("Context:", with_ctx)
        self.assertNotIn("Context:", without_ctx)
        for shared in (SYSTEM_PERSONA, ABSTENTION_INSTRUCTION, f"Question: {QUERY}", "Answer:"):
            self.assertIn(shared, with_ctx)
            self.assertIn(shared, without_ctx)

    def test_v1_bare_prompt_is_gone(self):
        self.assertNotEqual(build_answer_prompt(QUERY, None), f"Question: {QUERY}\nAnswer:")

    def test_empty_context_is_treated_as_no_rag(self):
        self.assertEqual(build_answer_prompt(QUERY, []), build_answer_prompt(QUERY, None))

    def test_all_context_chunks_appear(self):
        prompt = build_answer_prompt(QUERY, CONTEXT)
        for chunk in CONTEXT:
            self.assertIn(chunk, prompt)

    def test_query_appears_verbatim(self):
        self.assertIn(QUERY, build_answer_prompt(QUERY, CONTEXT))


class TestContextFormatting(unittest.TestCase):
    def test_chunks_are_blank_line_separated(self):
        self.assertEqual(format_context(["a", "b"]), "a\n\nb")

    def test_single_chunk(self):
        self.assertEqual(format_context(["only"]), "only")

    def test_empty(self):
        self.assertEqual(format_context([]), "")


class TestRoutingPrompt(unittest.TestCase):
    def test_demands_a_single_bare_word(self):
        # The parser is exact-match anchored, so the prompt must ask for a
        # bare token rather than merely "respond with yes or no".
        prompt = build_routing_prompt(QUERY).lower()
        self.assertIn("single word", prompt)
        self.assertIn("nothing else", prompt)


class TestSubQueryPrompt(unittest.TestCase):
    def test_requested_count_is_interpolated(self):
        # v1 asked for two, sliced two, and the paper claimed three. The count
        # must come from one place.
        self.assertIn("two", build_subquery_prompt(QUERY, n=2))
        self.assertIn("three", build_subquery_prompt(QUERY, n=3))

    def test_falls_back_to_digits_for_large_n(self):
        self.assertIn("7", build_subquery_prompt(QUERY, n=7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
