"""Tests for the four pipeline arms.

The generator and vector DB are stubs, so these run with no Ollama, no FAISS,
no model weights and no network.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.advanced.retriever import AdvancedRAGPipeline  # noqa: E402
from rag.modular.retriever import ModularRAGPipeline  # noqa: E402
from rag.naive.retriever import NaiveRAGPipeline  # noqa: E402
from rag.no_rag.retriever import NoRAGPipeline  # noqa: E402
from rag.prompts import ABSTENTION_INSTRUCTION, SYSTEM_PERSONA  # noqa: E402
from tests.test_timing import FakeClock  # noqa: E402


class StubGenerator:
    """Returns queued responses and records every prompt it received.

    ``call_seconds`` makes each call cost a fixed amount of a fake clock, so
    latency assertions are exact rather than timing-dependent. Passing a clock
    is what lets the tests prove the pipeline totals *all* of its calls.
    """

    def __init__(self, responses=None, default="A generated answer.",
                 clock=None, call_seconds=0.0, tokens=(50, 5)):
        self.responses = list(responses or [])
        self.default = default
        self.prompts = []
        self.clock = clock
        self.call_seconds = call_seconds
        self.prompt_tokens, self.generation_tokens = tokens

    def generate(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        if self.clock is not None:
            self.clock.advance(self.call_seconds)
        text = self.responses.pop(0) if self.responses else self.default
        return {
            "response": text,
            "error": False,
            "latency": self.call_seconds or 0.1,
            "generation_tokens": self.generation_tokens,
            "prompt_tokens": self.prompt_tokens,
        }


class StubDB:
    """Minimal VectorDBManager stand-in with deterministic scores."""

    def __init__(self, texts=None):
        texts = texts or ["chunk one text", "chunk two text", "chunk three text",
                          "chunk four text", "chunk five text"]
        self.doc_chunks = [{"text": t, "metadata": {"source": "s", "chunk_id": i}}
                           for i, t in enumerate(texts)]
        self.queries = []

    def search(self, query, top_k=3):
        self.queries.append((query, top_k))
        return [{"chunk": c, "score": 0.9 - 0.01 * i}
                for i, c in enumerate(self.doc_chunks[:top_k])]


class TestNoRAGPipeline(unittest.TestCase):
    def test_returns_empty_context(self):
        pipe = NoRAGPipeline(generator=StubGenerator())
        self.assertEqual(pipe.run("q")["retrieved_context"], [])

    def test_prompt_has_parity_with_rag_arms(self):
        gen = StubGenerator()
        NoRAGPipeline(generator=gen).run("What is a container?")
        prompt = gen.prompts[0]
        self.assertIn(SYSTEM_PERSONA, prompt)
        self.assertIn(ABSTENTION_INSTRUCTION, prompt)

    def test_does_not_retrieve(self):
        db = StubDB()
        NoRAGPipeline(db_manager=db, generator=StubGenerator()).run("q")
        self.assertEqual(db.queries, [])


class TestNaiveRAGPipeline(unittest.TestCase):
    def test_retrieves_top_k_and_returns_context(self):
        db, gen = StubDB(), StubGenerator()
        result = NaiveRAGPipeline(db, gen, top_k=3).run("q")
        self.assertEqual(len(result["retrieved_context"]), 3)
        self.assertEqual(db.queries, [("q", 3)])

    def test_context_appears_in_prompt(self):
        db, gen = StubDB(), StubGenerator()
        result = NaiveRAGPipeline(db, gen, top_k=2).run("q")
        for chunk in result["retrieved_context"]:
            self.assertIn(chunk, gen.prompts[0])

    def test_one_generation_call(self):
        gen = StubGenerator()
        NaiveRAGPipeline(StubDB(), gen).run("q")
        self.assertEqual(len(gen.prompts), 1)


class TestModularRouting(unittest.TestCase):
    """Guards v1 defect #7: `if "no" in decision` was a substring test."""

    def _route(self, router_reply):
        gen = StubGenerator(responses=[router_reply])
        pipe = ModularRAGPipeline(StubDB(), gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(StubDB(), gen))
        return pipe.route_query("q")[0]

    def test_plain_yes_routes_to_rag(self):
        self.assertEqual(self._route("yes"), "rag")

    def test_plain_no_routes_to_no_rag(self):
        self.assertEqual(self._route("no"), "no_rag")

    def test_affirmative_containing_the_substring_no(self):
        # Every one of these was silently misrouted to no_rag in v1.
        for reply in (
            "Yes, you need to know the document contents.",
            "Yes. There is no way to answer without the files.",
            "Yes, the answer cannot be given without retrieval.",
            "Yes - nothing else would suffice.",
            "Yes, I do not have this memorised.",
        ):
            self.assertEqual(self._route(reply), "rag", reply)

    def test_negative_with_trailing_explanation(self):
        self.assertEqual(self._route("No, this is general knowledge."), "no_rag")

    def test_case_and_whitespace_insensitive(self):
        for reply in ("  YES  ", "Yes\n", "**yes**", '"yes"'):
            self.assertEqual(self._route(reply), "rag", reply)
        for reply in ("  NO  ", "No\n", "**no**"):
            self.assertEqual(self._route(reply), "no_rag", reply)

    def test_unparseable_reply_defaults_to_rag(self):
        # Retrieval is the variable under study; an unreadable router response
        # must not silently turn a modular run into a baseline run.
        for reply in ("", "   ", "I'm not sure what you mean.", "banana"):
            self.assertEqual(self._route(reply), "rag", repr(reply))

    def test_raw_decision_is_recorded(self):
        gen = StubGenerator(responses=["Yes, definitely"])
        pipe = ModularRAGPipeline(StubDB(), gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(StubDB(), gen))
        self.assertEqual(pipe.route_query("q")[1], "Yes, definitely")

    def test_route_recorded_in_result(self):
        gen = StubGenerator(responses=["yes", "sub one\nsub two", "final answer"])
        pipe = ModularRAGPipeline(StubDB(), gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(StubDB(), gen))
        self.assertEqual(pipe.run("q")["route"], "rag")


class TestModularSubQueries(unittest.TestCase):
    def _subqueries(self, reply, n=2):
        gen = StubGenerator(responses=[reply])
        pipe = ModularRAGPipeline(StubDB(), gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(StubDB(), gen), n_sub_queries=n)
        return pipe.generate_sub_queries("original query")

    def test_plain_lines(self):
        self.assertEqual(self._subqueries("first query\nsecond query"),
                         ["first query", "second query"])

    def test_numbered_list_markers_stripped(self):
        self.assertEqual(self._subqueries("1. first query\n2. second query"),
                         ["first query", "second query"])

    def test_bullet_markers_stripped(self):
        self.assertEqual(self._subqueries("- first query\n* second query"),
                         ["first query", "second query"])

    def test_paren_numbering_stripped(self):
        self.assertEqual(self._subqueries("1) first query\n2) second query"),
                         ["first query", "second query"])

    def test_blank_lines_ignored(self):
        self.assertEqual(self._subqueries("first query\n\n\nsecond query"),
                         ["first query", "second query"])

    def test_duplicates_dropped(self):
        self.assertEqual(self._subqueries("same query\nsame query"), ["same query"])

    def test_echo_of_original_dropped(self):
        # A model that parrots the question must not add an identical
        # retrieval list to the fusion input.
        self.assertEqual(self._subqueries("original query\nreal sub query"),
                         ["real sub query"])

    def test_respects_requested_count(self):
        reply = "one\ntwo\nthree\nfour"
        self.assertEqual(len(self._subqueries(reply, n=2)), 2)
        self.assertEqual(len(self._subqueries(reply, n=3)), 3)

    def test_empty_reply_yields_no_subqueries(self):
        self.assertEqual(self._subqueries(""), [])


class TestModularRun(unittest.TestCase):
    def test_no_rag_route_delegates_and_returns_empty_context(self):
        gen = StubGenerator(responses=["no", "direct answer"])
        pipe = ModularRAGPipeline(StubDB(), gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(StubDB(), gen))
        result = pipe.run("q")
        self.assertEqual(result["route"], "no_rag")
        self.assertEqual(result["retrieved_context"], [])

    def test_rag_route_searches_original_plus_subqueries(self):
        db = StubDB()
        gen = StubGenerator(responses=["yes", "sub one\nsub two", "answer"])
        pipe = ModularRAGPipeline(db, gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(db, gen), top_k=3)
        pipe.run("original")
        searched = [q for q, _ in db.queries]
        self.assertEqual(searched, ["original", "sub one", "sub two"])

    def test_fused_context_respects_top_k(self):
        db = StubDB()
        gen = StubGenerator(responses=["yes", "sub one\nsub two", "answer"])
        pipe = ModularRAGPipeline(db, gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(db, gen), top_k=2)
        self.assertEqual(len(pipe.run("q")["retrieved_context"]), 2)

    def test_candidate_pool_is_wider_than_top_k(self):
        db = StubDB()
        gen = StubGenerator(responses=["yes", "sub one\nsub two", "answer"])
        pipe = ModularRAGPipeline(db, gen, NoRAGPipeline(generator=gen),
                                  NaiveRAGPipeline(db, gen), top_k=3,
                                  candidate_multiplier=2)
        pipe.run("q")
        self.assertTrue(all(k == 6 for _, k in db.queries))


class StubReranker:
    """Scores candidates in ascending order so reranking is observable."""

    def __init__(self):
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return [float(i) for i in range(len(pairs))]


class StubSparseIndex:
    """Word-overlap stand-in for TfidfIndex, so tests need no scikit-learn.

    Satisfies the same tiny protocol the pipeline depends on -- construction
    from a list of texts, plus ``scores(query)`` -- and counts how many times
    it was constructed so fit-once caching can be asserted directly.
    """

    instances = 0

    def __init__(self, texts):
        type(self).instances += 1
        self.texts = list(texts)
        self.token_sets = [set(t.lower().split()) for t in self.texts]

    def scores(self, query):
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return [0.0] * len(self.texts)
        return [len(query_tokens & tokens) / len(query_tokens) for tokens in self.token_sets]


class TestAdvancedPipeline(unittest.TestCase):
    def setUp(self):
        StubSparseIndex.instances = 0

    def _pipe(self, gen, db=None, **kw):
        kw.setdefault("sparse_index_factory", StubSparseIndex)
        return AdvancedRAGPipeline(db or StubDB(), gen, rerank=False, **kw)

    def test_query_rewrite_used_for_search(self):
        db = StubDB()
        gen = StubGenerator(responses=["optimized search terms", "answer"])
        self._pipe(gen, db).run("original question")
        self.assertEqual(db.queries[0][0], "optimized search terms")

    def test_rewrite_label_prefix_stripped(self):
        gen = StubGenerator(responses=["Optimized Query: clean terms", "answer"])
        self.assertEqual(self._pipe(gen).rewrite_query("q"), "clean terms")

    def test_rewrite_falls_back_to_original_on_empty(self):
        for reply in ("", "  ", "ab", "ERROR: model unavailable"):
            gen = StubGenerator(responses=[reply])
            self.assertEqual(self._pipe(gen).rewrite_query("original"), "original", repr(reply))

    def test_returns_top_k_contexts(self):
        gen = StubGenerator(responses=["rewritten", "answer"])
        result = self._pipe(gen, top_k=3).run("q")
        self.assertEqual(len(result["retrieved_context"]), 3)

    def test_rewritten_query_recorded(self):
        gen = StubGenerator(responses=["rewritten terms", "answer"])
        self.assertEqual(self._pipe(gen).run("q")["rewritten_query"], "rewritten terms")

    def test_reranker_receives_the_rewritten_query(self):
        rr = StubReranker()
        gen = StubGenerator(responses=["rewritten terms", "answer"])
        AdvancedRAGPipeline(StubDB(), gen, rerank=True, reranker=rr, top_k=2,
                            sparse_index_factory=StubSparseIndex).run("q")
        self.assertTrue(all(p[0] == "rewritten terms" for p in rr.pairs))

    def test_rerank_reorders_results(self):
        rr = StubReranker()  # ascending scores -> last candidate wins
        gen = StubGenerator(responses=["rewritten", "answer"])
        result = AdvancedRAGPipeline(StubDB(), gen, rerank=True, reranker=rr, top_k=1,
                                     sparse_index_factory=StubSparseIndex).run("q")
        self.assertEqual(len(result["retrieved_context"]), 1)

    def test_rerank_top_n_is_explicit(self):
        pipe = AdvancedRAGPipeline(StubDB(), StubGenerator(), rerank=False, rerank_top_n=12,
                                   sparse_index_factory=StubSparseIndex)
        self.assertEqual(pipe.rerank_top_n, 12)

    def test_sparse_index_built_once_across_queries(self):
        # v1 refit the whole corpus on every query, inside the untimed region.
        db = StubDB()
        pipe = self._pipe(StubGenerator(default="answer"), db)
        pipe.run("first")
        first_index = pipe._sparse_index
        pipe.run("second")
        pipe.run("third")
        self.assertIsNotNone(first_index)
        self.assertIs(pipe._sparse_index, first_index)
        self.assertEqual(StubSparseIndex.instances, 1)
        self.assertEqual(pipe._sparse_fit_count, 1)

    def test_index_rebuilt_when_a_chunk_is_added(self):
        db = StubDB()
        pipe = self._pipe(StubGenerator(default="answer"), db)
        pipe.run("q")
        original = pipe._sparse_index
        db.doc_chunks.append({"text": "a newly added chunk", "metadata": {}})
        pipe.run("q")
        self.assertIsNot(pipe._sparse_index, original)
        self.assertEqual(StubSparseIndex.instances, 2)

    def test_index_rebuilt_when_chunk_text_changes_without_changing_count(self):
        # Length alone is not a safe cache key: a reindexed corpus of the same
        # size would otherwise be scored against a stale vocabulary.
        db = StubDB()
        pipe = self._pipe(StubGenerator(default="answer"), db)
        pipe.run("q")
        db.doc_chunks[0] = {"text": "entirely different content", "metadata": {}}
        pipe.run("q")
        self.assertEqual(StubSparseIndex.instances, 2)

    def test_dense_only_fallback_when_sparse_index_unavailable(self):
        # The real TfidfIndex raises ImportError without scikit-learn. The run
        # must degrade to dense-only rather than crash.
        def failing_factory(texts):
            raise ImportError("No module named 'sklearn'")

        gen = StubGenerator(responses=["rewritten", "answer"])
        pipe = AdvancedRAGPipeline(StubDB(), gen, rerank=False, top_k=2,
                                   sparse_index_factory=failing_factory)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            result = pipe.run("q")
        self.assertEqual(len(result["retrieved_context"]), 2)
        self.assertIn("falling back to dense-only", captured.getvalue())

    def test_sparse_signal_can_promote_a_chunk_dense_search_missed(self):
        # StubDB only ever returns the first `top_k` chunks, so a lexically
        # perfect match further down the corpus is invisible to dense search.
        # It must still be *eligible*, and must win once the lexical arm
        # carries the majority of the weight.
        db = StubDB(texts=["alpha one", "alpha two", "alpha three",
                           "alpha four", "unmistakable lexical marker"])
        gen = StubGenerator(responses=["unmistakable lexical marker", "answer"])
        pipe = self._pipe(gen, db, top_k=1, rerank_top_n=4, dense_weight=0.3)
        result = pipe.run("q")
        self.assertEqual(result["retrieved_context"], ["unmistakable lexical marker"])

    def test_dense_ordering_survives_a_sparse_only_candidate(self):
        # Regression guard: a chunk missing from the dense results is placed at
        # the bottom of the dense *normalised* scale. Injecting a raw 0.0
        # before min-max instead would make it the minimum and squeeze the four
        # genuine dense scores (0.87-0.90) into the top 3% of the range,
        # flattening distinctions the vector store had actually made.
        db = StubDB(texts=["alpha one", "alpha two", "alpha three",
                           "alpha four", "unmistakable lexical marker"])
        gen = StubGenerator(responses=["unmistakable lexical marker", "answer"])
        pipe = self._pipe(gen, db, top_k=4, rerank_top_n=4, dense_weight=1.0)
        scored = pipe.hybrid_search("unmistakable lexical marker", top_k=4)
        by_text = {r["chunk"]["text"]: r["score"] for r in scored}
        # Pure dense weighting: the retrieved four must span the whole [0, 1]
        # scale, not huddle near the top.
        self.assertAlmostEqual(by_text["alpha one"], 1.0)
        self.assertAlmostEqual(by_text["alpha four"], 0.0)
        self.assertAlmostEqual(by_text["alpha two"], 2 / 3)
        self.assertAlmostEqual(by_text["alpha three"], 1 / 3)


class TestCostAccounting(unittest.TestCase):
    """Guards the v1 defect that made the cost/benefit comparison invalid.

    v1 reported ``latency`` as the elapsed time of the final ``generate()``
    call. Everything before it -- rewriting, routing, sub-query generation,
    retrieval, reranking -- was untimed, and the untimed portion grows with
    pipeline sophistication. On the v1 data this inverted the ordering: no-RAG
    (one LLM call, no retrieval) reported a 20.8s mean, advanced (two LLM calls
    plus hybrid retrieval and reranking) reported 8.4s.

    Each LLM call costs the fake clock exactly 1.0s here, so the totals below
    are arithmetic rather than timing-dependent.
    """

    def _arms(self, clock):
        gen = lambda responses=None: StubGenerator(  # noqa: E731
            responses=responses, clock=clock, call_seconds=1.0, default="answer")
        return gen

    def test_single_call_arm_reports_that_one_call(self):
        clock = FakeClock()
        gen = StubGenerator(clock=clock, call_seconds=1.0)
        result = NoRAGPipeline(generator=gen, clock=clock).run("q")
        self.assertAlmostEqual(result["latency"], 1.0)
        self.assertEqual(result["n_llm_calls"], 1)

    def test_advanced_arm_totals_both_of_its_llm_calls(self):
        clock = FakeClock()
        gen = StubGenerator(responses=["rewritten", "answer"], clock=clock, call_seconds=1.0)
        pipe = AdvancedRAGPipeline(StubDB(), gen, rerank=False, clock=clock,
                                   sparse_index_factory=StubSparseIndex)
        result = pipe.run("q")
        # v1 would have reported 1.0 -- the answer call alone.
        self.assertAlmostEqual(result["generation_latency_s"], 1.0)
        self.assertAlmostEqual(result["latency"], 2.0)
        self.assertEqual(result["n_llm_calls"], 2)

    def test_modular_arm_totals_all_three_of_its_llm_calls(self):
        clock = FakeClock()
        gen = StubGenerator(responses=["yes", "sub one\nsub two", "answer"],
                            clock=clock, call_seconds=1.0)
        db = StubDB()
        pipe = ModularRAGPipeline(db, gen, NoRAGPipeline(generator=gen, clock=clock),
                                  NaiveRAGPipeline(db, gen, clock=clock), clock=clock)
        result = pipe.run("q")
        self.assertAlmostEqual(result["generation_latency_s"], 1.0)
        self.assertAlmostEqual(result["latency"], 3.0)
        self.assertEqual(result["n_llm_calls"], 3)

    def test_sophistication_ordering_is_no_longer_inverted(self):
        # The central regression: with equal per-call cost, latency must be
        # non-decreasing in the number of calls an arm makes.
        latencies = {}

        clock = FakeClock()
        gen = StubGenerator(clock=clock, call_seconds=1.0)
        latencies["no_rag"] = NoRAGPipeline(generator=gen, clock=clock).run("q")["latency"]

        clock = FakeClock()
        gen = StubGenerator(clock=clock, call_seconds=1.0)
        latencies["naive"] = NaiveRAGPipeline(StubDB(), gen, clock=clock).run("q")["latency"]

        clock = FakeClock()
        gen = StubGenerator(responses=["rewritten", "answer"], clock=clock, call_seconds=1.0)
        latencies["advanced"] = AdvancedRAGPipeline(
            StubDB(), gen, rerank=False, clock=clock,
            sparse_index_factory=StubSparseIndex).run("q")["latency"]

        clock = FakeClock()
        gen = StubGenerator(responses=["yes", "a\nb", "answer"], clock=clock, call_seconds=1.0)
        db = StubDB()
        latencies["modular"] = ModularRAGPipeline(
            db, gen, NoRAGPipeline(generator=gen, clock=clock),
            NaiveRAGPipeline(db, gen, clock=clock), clock=clock).run("q")["latency"]

        self.assertLessEqual(latencies["no_rag"], latencies["naive"])
        self.assertLess(latencies["naive"], latencies["advanced"])
        self.assertLess(latencies["advanced"], latencies["modular"])

    def test_reranking_cost_is_charged_to_the_arm_that_pays_it(self):
        clock = FakeClock()

        class SlowReranker:
            def predict(self, pairs):
                clock.advance(5.0)
                return [float(i) for i in range(len(pairs))]

        gen = StubGenerator(responses=["rewritten", "answer"], clock=clock, call_seconds=1.0)
        pipe = AdvancedRAGPipeline(StubDB(), gen, rerank=True, reranker=SlowReranker(),
                                   clock=clock, sparse_index_factory=StubSparseIndex)
        result = pipe.run("q")
        self.assertAlmostEqual(result["stage_rerank_s"], 5.0)
        self.assertAlmostEqual(result["latency"], 7.0)  # 2 calls + rerank

    def test_token_cost_counts_every_call_not_just_the_last(self):
        clock = FakeClock()
        gen = StubGenerator(responses=["rewritten", "answer"], clock=clock,
                            call_seconds=1.0, tokens=(50, 5))
        pipe = AdvancedRAGPipeline(StubDB(), gen, rerank=False, clock=clock,
                                   sparse_index_factory=StubSparseIndex)
        result = pipe.run("q")
        # v1 recorded only the final call, so the rewrite prompt was free.
        self.assertEqual(result["prompt_tokens"], 50)
        self.assertEqual(result["total_prompt_tokens"], 100)
        self.assertEqual(result["total_generation_tokens"], 10)

    def test_routed_modular_run_still_charges_its_router_call(self):
        # A modular run that routes to no_rag has still paid for the routing
        # call. v1 returned the delegate's record wholesale, so that call was
        # free -- flattering precisely the arm that made the extra request.
        clock = FakeClock()
        gen = StubGenerator(responses=["no", "direct answer"], clock=clock, call_seconds=1.0)
        db = StubDB()
        pipe = ModularRAGPipeline(db, gen, NoRAGPipeline(generator=gen, clock=clock),
                                  NaiveRAGPipeline(db, gen, clock=clock), clock=clock)
        result = pipe.run("q")
        self.assertEqual(result["route"], "no_rag")
        self.assertAlmostEqual(result["latency"], 2.0)  # router + delegated answer
        self.assertEqual(result["n_llm_calls"], 2)
        self.assertEqual(result["total_prompt_tokens"], 100)

    def test_failed_call_still_reports_its_wall_clock_cost(self):
        clock = FakeClock()

        class FailingGenerator:
            def generate(self, prompt, system_prompt=None):
                clock.advance(2.0)
                return {"response": None, "error": True,
                        "error_message": "ConnectionError: refused",
                        "latency": 2.0, "prompt_tokens": None, "generation_tokens": None}

        result = NoRAGPipeline(generator=FailingGenerator(), clock=clock).run("q")
        self.assertTrue(result["error"])
        self.assertAlmostEqual(result["latency"], 2.0)
        self.assertEqual(result["failed_llm_calls"], 1)
        # An unmeasured token count is None, not zero: zeros were averaged into
        # v1's throughput means and dragged them toward zero.
        self.assertIsNone(result["total_prompt_tokens"])

    def test_stage_breakdown_accounts_for_the_whole_run(self):
        clock = FakeClock()
        gen = StubGenerator(responses=["rewritten", "answer"], clock=clock, call_seconds=1.0)
        pipe = AdvancedRAGPipeline(StubDB(), gen, rerank=False, clock=clock,
                                   sparse_index_factory=StubSparseIndex)
        result = pipe.run("q")
        self.assertAlmostEqual(result["unaccounted_latency_s"], 0.0, places=9)
        self.assertAlmostEqual(
            result["stage_rewrite_s"] + result["stage_retrieve_s"] + result["stage_generate_s"],
            result["latency"], places=9)


class TestAllArmsShareOnePromptShape(unittest.TestCase):

    def test_every_arm_uses_the_shared_persona_and_instruction(self):
        db = StubDB()
        plans = {
            "no_rag": ["answer"],
            "naive": ["answer"],
            "advanced": ["rewritten", "answer"],
            "modular": ["yes", "sub one\nsub two", "answer"],
        }
        for name, responses in plans.items():
            gen = StubGenerator(responses=list(responses))
            if name == "no_rag":
                NoRAGPipeline(generator=gen).run("q")
            elif name == "naive":
                NaiveRAGPipeline(db, gen).run("q")
            elif name == "advanced":
                AdvancedRAGPipeline(db, gen, rerank=False,
                                    sparse_index_factory=StubSparseIndex).run("q")
            else:
                ModularRAGPipeline(db, gen, NoRAGPipeline(generator=gen),
                                   NaiveRAGPipeline(db, gen)).run("q")
            answer_prompt = gen.prompts[-1]
            self.assertIn(SYSTEM_PERSONA, answer_prompt, name)
            self.assertIn(ABSTENTION_INSTRUCTION, answer_prompt, name)
            self.assertIn("Question: q", answer_prompt, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
