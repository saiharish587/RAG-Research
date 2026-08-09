"""No-RAG baseline: direct generation with no retrieval.

The control condition. Its prompt is produced by the same
:func:`~rag.prompts.build_answer_prompt` used by every RAG arm, differing only
in that no context block is present. In v1 this arm used a bare
``"Question: {q}\\nAnswer:"`` with no persona and no abstention instruction,
which confounded retrieval with prompt engineering: the measured no-RAG/RAG gap
could not be attributed to retrieval alone.
"""

from __future__ import annotations

import time
from typing import Any

from rag.base import RunRecorder
from rag.prompts import build_answer_prompt


class NoRAGPipeline:
    """Generates an answer directly from the query."""

    name = "no_rag"

    def __init__(self, db_manager=None, generator=None, clock=time.perf_counter):
        # db_manager is accepted but unused, so every arm shares one
        # construction signature.
        self.db_manager = db_manager
        self.generator = generator
        # Injectable so latency accounting can be asserted exactly in tests.
        self.clock = clock

    def run(self, query: str) -> dict[str, Any]:
        """Answer without retrieval.

        ``retrieved_context`` is an empty list, which the evaluator reads as
        "retrieval did not run" and reports as ``None`` for context-dependent
        metrics rather than as a zero.
        """
        recorder = RunRecorder(clock=self.clock)
        prompt = build_answer_prompt(query, retrieved_context=None)
        final = recorder.generate(self.generator, prompt, stage="generate")
        return recorder.finish(final, retrieved_context=[])
