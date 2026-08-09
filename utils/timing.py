"""Stage-level wall-clock accounting for a single pipeline run.

Why this exists
---------------
v1 reported ``latency`` as the elapsed time of the *final* ``generate()`` call,
because a pipeline's ``run()`` returned the generator's own result dict and the
driver read ``latency`` straight off it. Everything before that call was
untimed: query rewriting, routing, sub-query generation, dense search, TF-IDF
scoring and cross-encoder reranking.

The amount of untimed work grows with pipeline sophistication, so the omission
is directional rather than noisy. On the v1 data it inverted the ordering it was
meant to establish -- the no-RAG arm, which makes one LLM call and no retrieval,
reported a 20.8s mean while the advanced arm, which makes two LLM calls plus
hybrid retrieval and reranking, reported 8.4s.

Design
------
Stages must be siblings, never nested: a nested stage would be counted twice,
once in its own total and once inside its parent. Rather than trusting that
convention, :meth:`StageTimer.stage` raises on re-entry, so a double-counting
bug fails loudly in the tests instead of quietly biasing a published number.

The clock is :func:`time.perf_counter` -- monotonic and the highest resolution
available. :func:`time.time` is wall-clock and can step backwards when NTP
adjusts it, which would produce negative durations.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    """Accumulates non-overlapping named stages within one run.

    Repeated stages accumulate, so a pipeline issuing three retrievals reports
    one ``retrieve`` total and a count of 3. ``elapsed`` measures the whole run
    independently of the stages, which makes unaccounted overhead visible
    instead of silently absorbed.
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._start = clock()
        self._active: str | None = None
        self.stages: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        """Time a block and attribute it to ``name``."""
        if self._active is not None:
            raise RuntimeError(
                f"stage {name!r} started inside stage {self._active!r}; "
                "stages must not nest or their time is counted twice"
            )
        self._active = name
        start = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - start
            self._active = None
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    @property
    def elapsed(self) -> float:
        """Seconds since this timer was created."""
        return self._clock() - self._start

    @property
    def accounted(self) -> float:
        """Total time attributed to named stages."""
        return sum(self.stages.values())

    def unaccounted(self) -> float:
        """Elapsed time not attributed to any stage.

        Small values are ordinary interpreter overhead. A large value means a
        real cost is being left out of the reported total.
        """
        return self.elapsed - self.accounted

    def report(self, prefix: str = "stage_") -> dict[str, float | int]:
        """Flat, CSV-friendly view of the breakdown."""
        report: dict[str, float | int] = {}
        for name, seconds in self.stages.items():
            report[f"{prefix}{name}_s"] = seconds
            report[f"{prefix}{name}_calls"] = self.counts[name]
        report["pipeline_latency_s"] = self.elapsed
        report["unaccounted_latency_s"] = self.unaccounted()
        return report
