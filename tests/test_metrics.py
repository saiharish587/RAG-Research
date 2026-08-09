"""Tests for the pure metric functions.

These run on the standard library alone: no GPU, no model weights, no network.
Each test that guards a v1 defect names the defect, so a future change that
reintroduces it fails with an explanation rather than a bare assertion error.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import (  # noqa: E402
    ResponseStatus,
    classify_response,
    content_tokens,
    context_precision,
    context_recall,
    cosine_similarity,
    exact_match,
    groundedness,
    is_refusal,
    mean_ci,
    normalise_text,
    retrieval_hit_rate,
    token_f1,
    tokenise,
)


class TestNormalisation(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalise_text("The Bell-LaPadula Model."), "belllapadula model")

    def test_drops_articles(self):
        self.assertEqual(normalise_text("a cat and the dog"), "cat and dog")

    def test_collapses_whitespace(self):
        self.assertEqual(normalise_text("  spaced \n\t out  "), "spaced out")

    def test_handles_none_and_empty(self):
        self.assertEqual(normalise_text(""), "")

    def test_unicode_is_folded(self):
        # OCR'd slides contain typographic quotes; they must not survive as
        # distinct tokens or identical text would fail to match.
        self.assertEqual(normalise_text("“DevOps”"), normalise_text('"DevOps"'))

    def test_content_tokens_drop_stopwords(self):
        self.assertEqual(content_tokens("this is the way of it"), ["way"])


class TestClassifyResponse(unittest.TestCase):
    """Guards v1 defect #1: empty responses scored as 0.0 accuracy."""

    def test_normal_response_is_ok(self):
        self.assertIs(classify_response("Docker isolates processes."), ResponseStatus.OK)

    def test_none_is_empty(self):
        self.assertIs(classify_response(None), ResponseStatus.EMPTY)

    def test_blank_string_is_empty(self):
        self.assertIs(classify_response(""), ResponseStatus.EMPTY)

    def test_whitespace_only_is_empty_not_ok(self):
        # The exact v1 failure: 870 rows generated thousands of whitespace
        # tokens and were scored as maximally wrong answers.
        self.assertIs(classify_response("   \n\n \t  "), ResponseStatus.EMPTY)

    def test_empty_takes_precedence_over_truncation(self):
        # A whitespace loop that ran to the decode ceiling is reported as
        # EMPTY: the salient fact is that no answer was produced.
        status = classify_response("\n\n\n   ", generation_tokens=4096, token_limit=4096)
        self.assertIs(status, ResponseStatus.EMPTY)

    def test_degenerate_loop_detected(self):
        self.assertIs(classify_response("the " * 200), ResponseStatus.DEGENERATE)

    def test_short_repetitive_answer_is_not_degenerate(self):
        # Guards against over-eager classification of legitimate short answers.
        self.assertIs(classify_response("Yes. Yes, it does."), ResponseStatus.OK)

    def test_long_legitimate_answer_is_not_degenerate(self):
        text = " ".join(f"token{i}" for i in range(200))
        self.assertIs(classify_response(text), ResponseStatus.OK)

    def test_truncation_detected(self):
        status = classify_response("A long partial answer", generation_tokens=512, token_limit=512)
        self.assertIs(status, ResponseStatus.TRUNCATED)

    def test_error_flag_wins(self):
        self.assertIs(classify_response("anything", error=True), ResponseStatus.ERROR)

    def test_is_failure_property(self):
        self.assertFalse(ResponseStatus.OK.is_failure)
        for status in (ResponseStatus.EMPTY, ResponseStatus.DEGENERATE,
                       ResponseStatus.TRUNCATED, ResponseStatus.ERROR):
            self.assertTrue(status.is_failure, status)


class TestRefusalDetection(unittest.TestCase):
    def test_contraction_is_detected(self):
        # Regression: patterns are matched post-normalisation, which strips
        # apostrophes. A pattern containing "n't" can never fire.
        self.assertTrue(is_refusal("I don't know."))
        self.assertTrue(is_refusal("The context doesn't contain that."))

    def test_common_abstentions(self):
        for text in (
            "I do not know",
            "The context does not contain enough information.",
            "There is not enough information to answer.",
            "I cannot determine the answer.",
            "I can't answer that.",
            "Insufficient context.",
            "No relevant information was found.",
            "Unable to determine.",
            "That is not specified in the provided context.",
        ):
            self.assertTrue(is_refusal(text), text)

    def test_real_answers_are_not_refusals(self):
        for text in (
            "The Bell-LaPadula model enforces no-read-up.",
            "Docker containers provide process isolation.",
            "CI/CD pipelines do contain automated testing stages.",
            "The three information security principles are confidentiality, integrity and availability.",
        ):
            self.assertFalse(is_refusal(text), text)

    def test_empty_is_not_a_refusal(self):
        # An empty generation is a failure, not a deliberate abstention.
        self.assertFalse(is_refusal(""))
        self.assertFalse(is_refusal(None))


class TestTokenF1(unittest.TestCase):
    def test_identical_text_scores_one(self):
        self.assertAlmostEqual(token_f1("the cat sat", "the cat sat"), 1.0)

    def test_disjoint_text_scores_zero(self):
        self.assertEqual(token_f1("alpha beta", "gamma delta"), 0.0)

    def test_partial_overlap(self):
        # Note "a" is an article and is dropped by normalisation, so this
        # compares 2 tokens against 2: overlap=1 -> p=r=1/2 -> f1=1/2.
        self.assertAlmostEqual(token_f1("a b c", "a b d"), 0.5)

    def test_partial_overlap_without_articles(self):
        # pred=3 gold=3 overlap=2 -> p=r=2/3 -> f1=2/3
        self.assertAlmostEqual(token_f1("alpha beta gamma", "alpha beta delta"), 2 / 3)

    def test_repetition_cannot_inflate_score(self):
        # Multiset intersection: repeating a correct token must not raise
        # recall above what a single occurrence earns.
        padded = token_f1("cat cat cat cat", "cat dog")
        self.assertLess(padded, token_f1("cat", "cat dog"))

    def test_empty_inputs_score_zero(self):
        self.assertEqual(token_f1("", "something"), 0.0)
        self.assertEqual(token_f1("something", ""), 0.0)

    def test_normalisation_applies(self):
        self.assertAlmostEqual(token_f1("The Cat!", "cat"), 1.0)

    def test_range_is_wider_than_cosine(self):
        # The point of adopting F1: it must actually separate a good answer
        # from an off-topic one, which cosine (0.746-0.950 observed) did not.
        good = token_f1("docker isolates processes using namespaces",
                        "docker isolates processes using namespaces")
        bad = token_f1("the weather is pleasant today",
                       "docker isolates processes using namespaces")
        self.assertGreater(good - bad, 0.9)


class TestExactMatch(unittest.TestCase):
    def test_match_after_normalisation(self):
        self.assertEqual(exact_match("The Cat.", "cat"), 1.0)

    def test_mismatch(self):
        self.assertEqual(exact_match("cat", "dog"), 0.0)


class TestGroundedness(unittest.TestCase):
    """Guards v1 defect #2: hallucination_rate == 1 - accuracy."""

    def test_fully_supported_response(self):
        ctx = ["Kubernetes orchestrates containers across a cluster."]
        self.assertAlmostEqual(groundedness("Kubernetes orchestrates containers", ctx), 1.0)

    def test_unsupported_response(self):
        ctx = ["Kubernetes orchestrates containers."]
        self.assertEqual(groundedness("Photosynthesis converts sunlight", ctx), 0.0)

    def test_partially_supported(self):
        ctx = ["Kubernetes orchestrates containers."]
        # content tokens: kubernetes(y) orchestrates(y) photosynthesis(n) -> 2/3
        self.assertAlmostEqual(
            groundedness("Kubernetes orchestrates photosynthesis", ctx), 2 / 3
        )

    def test_none_without_context(self):
        # Critical: must NOT be 0.0. Scoring the no-RAG arm as fully
        # hallucinated is the error class that produced the v1 headline.
        self.assertIsNone(groundedness("Any answer at all", None))
        self.assertIsNone(groundedness("Any answer at all", []))

    def test_independent_of_ground_truth(self):
        # Groundedness must be computable without the gold answer; if it were
        # a function of ground truth it would just be accuracy again.
        ctx = ["Kubernetes orchestrates containers."]
        self.assertAlmostEqual(groundedness("Kubernetes orchestrates containers", ctx), 1.0)

    def test_stopword_only_response_returns_none(self):
        self.assertIsNone(groundedness("the of and is", ["Kubernetes orchestrates."]))


class TestRetrievalMetrics(unittest.TestCase):
    def test_context_precision_and_recall(self):
        ctx = ["alpha beta gamma"]
        gt = "alpha beta"
        self.assertAlmostEqual(context_precision(ctx, gt), 2 / 3)
        self.assertAlmostEqual(context_recall(ctx, gt), 1.0)

    def test_none_without_context(self):
        for fn in (context_precision, context_recall, retrieval_hit_rate):
            self.assertIsNone(fn(None, "gold"), fn.__name__)
            self.assertIsNone(fn([], "gold"), fn.__name__)

    def test_hit_rate_finds_relevant_chunk(self):
        gold = "alpha beta gamma delta"
        ctx = ["totally unrelated filler", "alpha beta gamma delta"]
        self.assertEqual(retrieval_hit_rate(ctx, gold), 1.0)

    def test_hit_rate_zero_when_nothing_relevant(self):
        self.assertEqual(
            retrieval_hit_rate(["unrelated filler text"], "alpha beta gamma delta"), 0.0
        )


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_result_is_clamped(self):
        self.assertLessEqual(cosine_similarity([1e-8, 1e-8], [1e-8, 1e-8]), 1.0)


class TestMeanCI(unittest.TestCase):
    def test_basic_statistics(self):
        stats = mean_ci([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["sd"], math.sqrt(5 / 3))

    def test_interval_brackets_the_mean(self):
        stats = mean_ci([1.0, 2.0, 3.0, 4.0])
        self.assertLess(stats["ci_low"], stats["mean"])
        self.assertGreater(stats["ci_high"], stats["mean"])

    def test_none_values_are_skipped(self):
        # None means "undefined", which must not be coerced to zero.
        self.assertAlmostEqual(mean_ci([1.0, None, 3.0])["mean"], 2.0)

    def test_empty_input(self):
        stats = mean_ci([])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["mean"])

    def test_single_value_has_zero_width_interval(self):
        stats = mean_ci([5.0])
        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["ci_low"], 5.0)
        self.assertEqual(stats["ci_high"], 5.0)

    def test_wider_confidence_gives_wider_interval(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        narrow = mean_ci(data, confidence=0.95)
        wide = mean_ci(data, confidence=0.99)
        self.assertGreater(wide["ci_high"] - wide["ci_low"], narrow["ci_high"] - narrow["ci_low"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
