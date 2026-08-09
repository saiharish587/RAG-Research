"""Aggregation of per-run rows into reportable statistics.

The v1 pipeline aggregated with a bare ``groupby(...).mean()`` over every
column, which is what allowed three separate errors to reach the paper:

* failures averaged into quality means as zeros;
* means reported with no dispersion (coefficients of variation up to 535%);
* 30 deterministic repeats counted as 30 independent samples, inflating
  apparent significance roughly 5.5-fold.

This module reports quality **only over successful generations**, always with
n, sd and a confidence interval, and separates the two units of analysis:

``per_run``
    Every row. The correct unit for *cost* metrics (latency, tokens), which
    genuinely vary between repeats.
``per_query``
    Rows collapsed to one value per query first. The correct unit for
    *quality* metrics: with ``temperature=0`` and a fixed seed the repeats of a
    single query are the same measurement made 30 times, so the independent
    sample size is the number of questions (13), not the number of rows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from evaluation.metrics import mean_ci

__all__ = ["failure_breakdown", "aggregate_quality", "aggregate_cost", "summarise"]

QUALITY_METRICS = (
    "token_f1",
    "exact_match",
    "cosine",
    "groundedness",
    "context_precision",
    "context_recall",
    "retrieval_hit_rate",
)

COST_METRICS = ("latency", "generation_tokens", "prompt_tokens")


def failure_breakdown(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Generation failure rate with its denominator and a per-status split.

    Reported alongside quality, never inside it. A configuration that answers
    8% of the time at 90% quality and one that answers 100% of the time at 7%
    quality are different findings; v1 could not tell them apart.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "n_failed": 0, "failure_rate": None, "by_status": {}}

    by_status: dict[str, int] = defaultdict(int)
    for row in rows:
        by_status[row.get("status", "ok")] += 1

    n_failed = sum(v for k, v in by_status.items() if k != "ok")
    n_refusal = sum(1 for r in rows if r.get("is_refusal"))
    return {
        "n": n,
        "n_failed": n_failed,
        "failure_rate": n_failed / n,
        "n_refusal": n_refusal,
        "refusal_rate": n_refusal / n,
        "by_status": dict(by_status),
    }


def _successful(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("is_failure", r.get("status", "ok") != "ok")]


def aggregate_quality(
    rows: Sequence[dict[str, Any]],
    *,
    group_by_query: bool = True,
    query_key: str = "query",
) -> dict[str, dict[str, Any]]:
    """Quality statistics over successful generations only.

    With ``group_by_query`` (the default) each query contributes a single
    averaged value before the statistics are taken, so ``n`` is the number of
    distinct questions. This is the honest unit of analysis for a deterministic
    decode and it is what any significance test downstream must use.
    """
    ok_rows = _successful(rows)
    out: dict[str, dict[str, Any]] = {}

    for metric in QUALITY_METRICS:
        if group_by_query:
            buckets: dict[Any, list[float]] = defaultdict(list)
            for row in ok_rows:
                value = row.get(metric)
                if value is not None:
                    buckets[row.get(query_key)].append(float(value))
            values = [sum(v) / len(v) for v in buckets.values() if v]
        else:
            values = [float(r[metric]) for r in ok_rows if r.get(metric) is not None]

        out[metric] = mean_ci(values)
        out[metric]["unit"] = "query" if group_by_query else "run"
    return out


def aggregate_cost(rows: Sequence[dict[str, Any]], *, include_failures: bool = True) -> dict[str, dict[str, Any]]:
    """Cost statistics (latency, tokens) at the per-run unit.

    Failures are included by default and deliberately: a run that looped on
    whitespace for 44 seconds really did cost 44 seconds, and excluding it
    would understate the cost of an unreliable configuration.

    Both mean and median are returned. The v1 "Reranking Efficiency Paradox"
    was a single 226.8 s outlier in one cell of 390 rows; under the median the
    effect reverses sign, so the median is reported next to the mean by default
    rather than on request.
    """
    source = rows if include_failures else _successful(rows)
    out: dict[str, dict[str, Any]] = {}
    for metric in COST_METRICS:
        values = [float(r[metric]) for r in source if r.get(metric) is not None]
        stats = mean_ci(values)
        stats["median"] = _median(values)
        stats["unit"] = "run"
        out[metric] = stats
    return out


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarise(
    rows: Sequence[dict[str, Any]],
    *,
    group_by: Sequence[str] = ("model", "rag_type"),
    query_key: str = "query",
) -> dict[tuple, dict[str, Any]]:
    """Full summary per configuration: failures, quality and cost together.

    Returned as ``{(model, rag_type): {"failures": ..., "quality": ...,
    "cost": ...}}``. Presenting the three side by side is the structural fix
    for v1's headline error -- a quality number is not interpretable without
    the failure rate that produced its denominator.
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_by)].append(row)

    return {
        key: {
            "failures": failure_breakdown(group),
            "quality": aggregate_quality(group, query_key=query_key),
            "cost": aggregate_cost(group),
        }
        for key, group in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]])
    }
