"""Ollama generation wrapper with honest cost accounting.

Three v1 defects are corrected here.

**Errors were indistinguishable from answers.** A failed call returned the
string ``"ERROR: <exception>"`` in the same ``response`` field as a real
answer, with no flag. Downstream that text was scored as though the model had
said it: it was embedded, compared to ground truth, and averaged into quality
means. An explicit ``error`` boolean is now returned and the evaluator keys off
it.

**Failed calls reported zero cost.** ``prompt_eval_speed`` and
``generation_speed`` were set to ``0.0`` on error, and those zeros were then
averaged into throughput means, dragging them toward zero in proportion to the
failure rate -- which differs by arm. They are now ``None``, so they are
excluded from means rather than deflating them.

**Model load time was inside the reported latency.** ``load_duration`` covers
pulling weights into memory and is paid once per model, not once per query.
Left inside ``latency`` it inflates whichever arm happens to run first. It is
now reported separately and a warmup call absorbs it before measurement begins.
"""

from __future__ import annotations

import time
from typing import Any

# Ollama reports all durations in nanoseconds.
_NS_PER_S = 1e9


class Generator:
    """Single-turn generation against an Ollama model."""

    def __init__(self, model_name: str = "qwen2.5:0.5b", temperature: float = 0.0,
                 seed: int = 42, client=None):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        # Injectable so the calling code is testable without an Ollama daemon.
        self._client = client
        self._warmed_up = False

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama
        return self._client

    # ------------------------------------------------------------------

    def warmup(self, prompt: str = "hello") -> float:
        """Force the model into memory before any measured call.

        Returns the seconds spent. The first request to a cold model pays the
        weight-loading cost; charging that to whichever query happened to be
        first would attribute a per-model cost to a single arm.
        """
        start = time.perf_counter()
        self.generate(prompt)
        self._warmed_up = True
        return time.perf_counter() - start

    def generate(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        """Generate once and report what it cost.

        Returns a dict with ``response``, ``error``, token counts, ``latency``
        (wall clock for this call), ``load_duration`` (reported separately, not
        subtracted) and throughput figures that are ``None`` when undefined.
        """
        start = time.perf_counter()
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_prompt or "",
                options={"temperature": self.temperature, "seed": self.seed},
            )
        except Exception as exc:
            print(f"Error during generation with model {self.model_name}: {exc}")
            return {
                "response": None,
                "error": True,
                "error_message": f"{type(exc).__name__}: {exc}",
                "prompt_tokens": None,
                "generation_tokens": None,
                # The wall-clock cost of a failure is real and is recorded.
                "latency": time.perf_counter() - start,
                "load_duration": None,
                # Throughput is undefined, not zero. Zeros here were averaged
                # into v1's speed means and pulled them toward zero.
                "prompt_eval_speed": None,
                "generation_speed": None,
                "warmed_up": self._warmed_up,
                "raw_stats": {},
            }

        latency = time.perf_counter() - start

        load_duration = _seconds(response.get("load_duration"))
        prompt_tokens = response.get("prompt_eval_count")
        generation_tokens = response.get("eval_count")
        prompt_eval_duration = _seconds(response.get("prompt_eval_duration"))
        eval_duration = _seconds(response.get("eval_duration"))

        return {
            "response": (response.get("response") or "").strip(),
            "error": False,
            "error_message": None,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "latency": latency,
            # Reported alongside latency rather than folded into it, so a cold
            # start is visible instead of being charged to one arm's mean.
            "load_duration": load_duration,
            "prompt_eval_speed": _rate(prompt_tokens, prompt_eval_duration),
            "generation_speed": _rate(generation_tokens, eval_duration),
            "warmed_up": self._warmed_up,
            "raw_stats": dict(response),
        }


def _seconds(nanoseconds) -> float | None:
    """Convert an Ollama nanosecond duration, preserving 'not reported'."""
    if nanoseconds is None:
        return None
    return nanoseconds / _NS_PER_S


def _rate(tokens, duration) -> float | None:
    """Tokens per second, or ``None`` where the rate is undefined.

    v1 returned ``0.0`` when the duration was zero or missing. Zero is a
    measurable throughput; an unmeasurable one is not, and averaging the two
    together understates the mean.
    """
    if not tokens or not duration or duration <= 0:
        return None
    return tokens / duration
