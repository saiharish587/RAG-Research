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
from typing import Any

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

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_query(self, query: str) -> tuple[str, str]:
        """Decide whether the query needs retrieval.

        Returns ``(route, raw_decision)`` where route is ``"rag"`` or
        ``"no_rag"``. Parsing is anchored at the start of the reply rather than
        searching for a substring anywhere in it.

        An unrecognised reply routes to ``"rag"``. Retrieval is the condition
        under study, so an unparseable router response must not silently
        convert a modular run into a baseline run.
        """
        response = self.generator.generate(build_routing_prompt(query))
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

    def generate_sub_queries(self, query: str) -> list[str]:
        """Ask the model for ``n_sub_queries`` alternative phrasings.

        Enumeration markers are stripped. Blank lines, duplicates and repeats
        of the original query are dropped, so a model that simply echoes the
        question does not inflate the fusion input with identical lists.
        """
        response = self.generator.generate(
            build_subquery_prompt(query, n=self.n_sub_queries)
        )
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
        route, raw_decision = self.route_query(query)
        if route == "no_rag":
            result = self.no_rag_pipeline.run(query)
            result["route"] = "no_rag"
            result["router_raw"] = raw_decision
            return result

        sub_queries = self.generate_sub_queries(query)
        all_queries = [query] + sub_queries

        retrieval_lists = [
            self.db_manager.search(q, top_k=self.top_k * self.candidate_multiplier)
            for q in all_queries
        ]

        fused_chunks = self.reciprocal_rank_fusion(retrieval_lists)
        retrieved_contexts = [c["text"] for c in fused_chunks]

        prompt = build_answer_prompt(query, retrieved_contexts)
        result = self.generator.generate(prompt)
        result["retrieved_context"] = retrieved_contexts
        result["route"] = "rag"
        result["router_raw"] = raw_decision
        result["sub_queries"] = sub_queries
        return result
