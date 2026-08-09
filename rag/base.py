"""Shared run recording for the four pipeline arms.

Every arm produces the same result record regardless of how many LLM calls,
retrievals or reranking passes it took to get there. That uniformity is what
makes the arms comparable: a cost metric that is defined differently per arm
cannot support a cost/benefit claim about arm choice.

What v1 got wrong
-----------------
``run()`` returned the generator's own dict for the *final* call, so the
driver's ``latency`` was the duration of that one call. Query rewriting,
routing, sub-query generation, retrieval and cross-encoder reranking were all
outside it. The untimed portion grows with pipeline sophistication, so the
omission was directional. On the v1 data it inverted the ordering it existed to
establish: no-RAG (one LLM call, no retrieval) reported a 20.8s mean while
advanced (two LLM calls plus hybrid retrieval and reranking) reported 8.4s.

Token cost had the same shape of error: only the final call's counts were
recorded, so the extra prompts the advanced and modular arms send were free in
the accounting.
"""

from __future__ import annotations

import time
from typing import Any

from utils.timing import StageTimer


class RunRecorder:
    """Times stages and accumulates the cost of every LLM call in a run."""

    def __init__(self, clock=time.perf_counter):
        self.timer = StageTimer(clock=clock)
        self.calls: list[dict[str, Any]] = []

    def stage(self, name: str):
        """Time a non-LLM stage (retrieval, reranking, fusion)."""
        return self.timer.stage(name)

    def generate(self, generator, prompt: str, stage: str,
                 system_prompt: str | None = None) -> dict[str, Any]:
        """Run one LLM call inside a timed stage and record its cost."""
        with self.timer.stage(stage):
            result = generator.generate(prompt, system_prompt)
        self.calls.append(result)
        return result

    # ------------------------------------------------------------------

    def finish(self, final: dict[str, Any], retrieved_context: list[str],
               **extra: Any) -> dict[str, Any]:
        """Assemble the run record.

        ``final`` is the answer-producing call; its status determines whether
        the run succeeded. An earlier call may have failed without failing the
        run -- a failed rewrite degrades to the original query, a failed router
        degrades to retrieval -- so those are counted in ``failed_llm_calls``
        rather than discarding an answer that was in fact produced.
        """
        record: dict[str, Any] = {
            "response": final.get("response"),
            "error": bool(final.get("error")),
            "error_message": final.get("error_message"),
            "retrieved_context": retrieved_context,
            "n_retrieved": len(retrieved_context),

            # Cost of the answer-producing call, comparable to v1's numbers.
            "prompt_tokens": final.get("prompt_tokens"),
            "generation_tokens": final.get("generation_tokens"),
            "generation_speed": final.get("generation_speed"),
            "prompt_eval_speed": final.get("prompt_eval_speed"),
            "generation_latency_s": final.get("latency"),

            # Cost of the run, which is what an arm actually charges you.
            "latency": self.timer.elapsed,
            "total_prompt_tokens": _total(self.calls, "prompt_tokens"),
            "total_generation_tokens": _total(self.calls, "generation_tokens"),
            "n_llm_calls": len(self.calls),
            "failed_llm_calls": sum(1 for c in self.calls if c.get("error")),
            # Weight loading is per-model, not per-query; kept separate so it
            # is never silently charged to whichever arm ran first.
            "load_duration": _total(self.calls, "load_duration"),
        }
        record.update(self.timer.report())
        record.update(extra)
        return record


def _total(calls: list[dict[str, Any]], field: str) -> float | int | None:
    """Sum a cost field across calls, ignoring the ones that never reported it.

    ``None`` when no call reported the field at all -- an unmeasured cost is
    not a cost of zero.
    """
    values = [c[field] for c in calls if c.get(field) is not None]
    return sum(values) if values else None
