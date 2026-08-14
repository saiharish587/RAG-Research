"""Oracle RAG: direct injection of gold ground-truth evidence context.

Serves as the theoretical upper bound for context utilization testing.
No vector DB retrieval or query transformation is performed; the ground-truth
context is injected directly into the generation prompt.
"""

from __future__ import annotations

import time
from typing import Any

from rag.base import RunRecorder
from rag.prompts import build_answer_prompt


class OracleRAGPipeline:
    """Injects gold ground-truth context directly into answer prompt."""

    name = "oracle"

    def __init__(self, generator, clock=time.perf_counter):
        self.generator = generator
        self.clock = clock

    def run(self, query: str, ground_truth_context: str | list[str]) -> dict[str, Any]:
        recorder = RunRecorder(clock=self.clock)

        if isinstance(ground_truth_context, str):
            retrieved_contexts = [ground_truth_context]
        else:
            retrieved_contexts = list(ground_truth_context or [])

        prompt = build_answer_prompt(query, retrieved_contexts)
        final = recorder.generate(self.generator, prompt, stage="generate")
        return recorder.finish(final, retrieved_contexts)
