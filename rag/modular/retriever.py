"""Modular RAG: query routing, sub-query expansion and rank fusion.

Two v1 defects are corrected here.

**Routing used a substring test.** ``if "no" in decision`` matched any reply
containing the letters "no" anywhere -- "know", "not", "cannot", "nothing" all
routed to no-RAG. Since the routing prompt asks whether retrieval is needed,
the most natural affirmative phrasings ("Yes, you need to know the document
contents") contain "no" and were silently misrouted. Measured on the v1 data,
between 7% and 46% of runs degraded to the no-RAG arm depending on model, so
the "modular" condition was partly measuring the baseline.

**Routing failures were indistinguishable from routing decisions.** An empty or
unparseable reply fell through to the same branch as a deliberate "no". The
route is now parsed explicitly and an unparseable reply falls back to
retrieval, with the raw decision recorded so routing behaviour is auditable
rather than inferred.
"""

from __future__ import annotations

import re
import time
from typing import Any

from rag.base import RunRecorder
from rag.fusion import RRF_K, reciprocal_rank_fusion
from rag.prompts import build_answer_prompt, build_routing_prompt, build_subquery_prompt

# Matches a leading enumeration marker: "1. ", "2) ", "- ", "* ".
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\s*[.)\]]|[-*•])\s*")

# An affirmative reply starts with yes/true/1; a negative starts with no/false/0.
# Anchored at the start so a trailing explanation cannot flip the decision.
_YES_RE = re.compile(r"^(?:yes|y|true|1)\b")
_NO_RE = re.compile(r"^(?:no|n|false|0)\b")


class ModularRAGPipeline:
    """Routes, expands into sub-queries, then fuses multiple retrieval lists."""

    name = "modular"

    def __init__(
        self,
        db_manager,
        generator,
        no_rag_pipeline,
        naive_rag_pipeline,
        top_k: int = 3,
        n_sub_queries: int = 2,
        candidate_multiplier: int = 2,
        rrf_k: int = RRF_K,
        clock=time.perf_counter,
    ):
        self.db_manager = db_manager
        self.generator = generator
        self.no_rag_pipeline = no_rag_pipeline
        self.naive_rag_pipeline = naive_rag_pipeline
        self.top_k = top_k
        # Number of sub-queries requested AND kept. One attribute drives both
        # the prompt and the slice, so they cannot disagree.
        self.n_sub_queries = n_sub_queries
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k = rrf_k
        self.clock = clock

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_query(self, query: str, recorder=None) -> tuple[str, str]:
        """Decide whether the query needs retrieval.

        Returns ``(route, raw_decision)`` where route is ``"rag"`` or
        ``"no_rag"``. Parsing is anchored at the start of the reply rather than
        searching for a substring anywhere in it.

        An unrecognised reply routes to ``"rag"``. Retrieval is the condition
        under study, so an unparseable router response must not silently
        convert a modular run into a baseline run.
        """
        prompt = build_routing_prompt(query)
        if recorder is None:
            response = self.generator.generate(prompt)
        else:
            response = recorder.generate(self.generator, prompt, stage="route")
        raw = (response.get("response") or "").strip()

        # Take the first word-ish token, ignoring markdown or quoting.
        decision = raw.lower().lstrip("*_`'\"“”‘’ \t\n")

        if _NO_RE.match(decision):
            return "no_rag", raw
        if _YES_RE.match(decision):
            return "rag", raw
        return "rag", raw

    # ------------------------------------------------------------------
    # Sub-queries
    # ------------------------------------------------------------------

    def generate_sub_queries(self, query: str, recorder=None) -> list[str]:
        """Ask the model for ``n_sub_queries`` alternative phrasings.

        Enumeration markers are stripped. Blank lines, duplicates and repeats
        of the original query are dropped, so a model that simply echoes the
        question does not inflate the fusion input with identical lists.
        """
        prompt = build_subquery_prompt(query, n=self.n_sub_queries)
        if recorder is None:
            response = self.generator.generate(prompt)
        else:
            response = recorder.generate(self.generator, prompt, stage="subquery")
        text = (response.get("response") or "").strip()

        seen = {query.strip().lower()}
        sub_queries: list[str] = []
        for line in text.split("\n"):
            cleaned = _LIST_MARKER_RE.sub("", line).strip()
            if not cleaned:
                continue
            if cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            sub_queries.append(cleaned)
            if len(sub_queries) == self.n_sub_queries:
                break
        return sub_queries

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def reciprocal_rank_fusion(self, list_of_retrievals, k: int | None = None) -> list[dict]:
        """Fuse retrieval lists by RRF and return the top-k chunks.

        Delegates to :func:`rag.fusion.reciprocal_rank_fusion`, which uses a
        1-based rank. v1 enumerated from 0, so the top hit scored ``1/60``
        instead of ``1/61``.
        """
        fused = reciprocal_rank_fusion(
            list_of_retrievals,
            key=lambda r: r["chunk"]["text"],
            k=self.rrf_k if k is None else k,
            top_n=self.top_k,
        )
        return [item["chunk"] for item, _score in fused]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict[str, Any]:
        recorder = RunRecorder(clock=self.clock)

        route, raw_decision = self.route_query(query, recorder=recorder)
        if route == "no_rag":
            # Delegate for the answer, but keep this run's cost accounting:
            # the routing call was paid by the modular arm and must be charged
            # to it. v1 returned the delegate's record wholesale, so a routed
            # run reported the baseline arm's cost and the router call was
            # free -- which flattered exactly the arm that made the extra call.
            with recorder.stage("delegate_no_rag"):
                delegated = self.no_rag_pipeline.run(query)
            return self._finish_delegated(recorder, delegated, route, raw_decision)

        sub_queries = self.generate_sub_queries(query, recorder=recorder)
        all_queries = [query] + sub_queries

        with recorder.stage("retrieve"):
            retrieval_lists = [
                self.db_manager.search(q, top_k=self.top_k * self.candidate_multiplier)
                for q in all_queries
            ]

        with recorder.stage("fuse"):
            fused_chunks = self.reciprocal_rank_fusion(retrieval_lists)
            retrieved_contexts = [c["text"] for c in fused_chunks]

        prompt = build_answer_prompt(query, retrieved_contexts)
        final = recorder.generate(self.generator, prompt, stage="generate")
        return recorder.finish(
            final,
            retrieved_contexts,
            route="rag",
            router_raw=raw_decision,
            sub_queries=sub_queries,
            n_sub_queries_used=len(sub_queries),
        )

    def _finish_delegated(self, recorder, delegated: dict[str, Any],
                          route: str, raw_decision: str) -> dict[str, Any]:
        """Merge a delegate's answer with this arm's own cost accounting."""
        record = recorder.finish(
            {
                "response": delegated.get("response"),
                "error": delegated.get("error"),
                "error_message": delegated.get("error_message"),
                "prompt_tokens": delegated.get("prompt_tokens"),
                "generation_tokens": delegated.get("generation_tokens"),
                "generation_speed": delegated.get("generation_speed"),
                "prompt_eval_speed": delegated.get("prompt_eval_speed"),
                "latency": delegated.get("generation_latency_s"),
            },
            retrieved_context=delegated.get("retrieved_context", []),
            route=route,
            router_raw=raw_decision,
            sub_queries=[],
            n_sub_queries_used=0,
        )
        # Add the delegate's own token cost to this run's totals; the recorder
        # only saw the routing call.
        for field in ("total_prompt_tokens", "total_generation_tokens"):
            record[field] = _add(record.get(field), delegated.get(field))
        record["n_llm_calls"] = record["n_llm_calls"] + delegated.get("n_llm_calls", 0)
        record["failed_llm_calls"] = (
            record["failed_llm_calls"] + delegated.get("failed_llm_calls", 0)
        )
        return record


def _add(a, b):
    """Sum two optional counters, preserving 'never measured' as ``None``."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b
