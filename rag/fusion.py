"""Rank and score fusion utilities.

Pure functions with no third-party dependencies, so the fusion arithmetic is
unit-testable without FAISS, an index or model weights.

Two v1 defects are corrected here:

1. **RRF used a 0-based rank.** The formula ``1/(k + rank)`` was evaluated with
   ``rank`` starting at 0, giving the top document a score of ``1/k`` instead
   of the standard ``1/(k+1)``. See :func:`reciprocal_rank_fusion`.

2. **Hybrid fusion mixed incomparable scales.** ``0.7 * dense + 0.3 * sparse``
   combined FAISS cosine (roughly 0.5-0.9 on this corpus) with TF-IDF cosine
   (typically 0.0-0.2), so the nominal 70/30 weighting was in practice close to
   pure dense retrieval. See :func:`weighted_score_fusion`.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Iterable, Sequence

__all__ = ["reciprocal_rank_fusion", "min_max_normalise", "weighted_score_fusion"]

# Standard RRF damping constant from Cormack et al. (2009).
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Any]],
    *,
    key: Callable[[Any], Hashable],
    k: int = RRF_K,
    top_n: int | None = None,
) -> list[tuple[Any, float]]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Score for one item is ``sum over lists of 1 / (k + rank)`` where ``rank`` is
    **1-based**. v1 enumerated from 0, so the first document scored ``1/60``
    rather than ``1/61``. The ordering within a single list is unaffected --
    which is why the bug was invisible -- but the *relative weight* of a rank-1
    hit against a rank-2 hit is distorted, so fusion across lists differs.

    Returns ``(item, score)`` pairs in descending score order. Ties are broken
    by best (lowest) rank achieved in any list, then by first appearance, so
    the result is deterministic and does not depend on dict ordering.
    """
    scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    first_seen: dict[Hashable, int] = {}
    items: dict[Hashable, Any] = {}
    counter = 0

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):  # 1-based: the fix
            identity = key(item)
            if identity not in items:
                items[identity] = item
                first_seen[identity] = counter
                best_rank[identity] = rank
                counter += 1
            else:
                best_rank[identity] = min(best_rank[identity], rank)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)

    ordered = sorted(
        scores,
        key=lambda i: (-scores[i], best_rank[i], first_seen[i]),
    )
    if top_n is not None:
        ordered = ordered[:top_n]
    return [(items[i], scores[i]) for i in ordered]


def min_max_normalise(values: Sequence[float]) -> list[float]:
    """Scale values to [0, 1].

    A constant input maps to all zeros rather than dividing by zero. Zero is
    chosen over 0.5 or 1.0 because a signal with no variation carries no
    ranking information and should not influence the fused order.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    spread = hi - lo
    if spread <= 0:
        return [0.0] * len(values)
    return [(v - lo) / spread for v in values]


def weighted_score_fusion(
    dense_scores: Sequence[float],
    sparse_scores: Sequence[float],
    *,
    dense_weight: float = 0.7,
    normalise: bool = True,
) -> list[float]:
    """Combine dense and sparse scores after putting them on a common scale.

    Both score vectors are min-max normalised before weighting, so
    ``dense_weight`` means what it says. Without normalisation the weights are
    applied to raw values whose ranges differ by roughly an order of magnitude,
    and the stated 70/30 blend silently behaves like ~95/5.

    ``sparse_weight`` is ``1 - dense_weight`` by construction, so the two
    cannot drift out of sync.
    """
    if len(dense_scores) != len(sparse_scores):
        raise ValueError(
            f"score vectors must align: {len(dense_scores)} dense vs {len(sparse_scores)} sparse"
        )
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError(f"dense_weight must be in [0, 1], got {dense_weight}")

    if normalise:
        dense = min_max_normalise(dense_scores)
        sparse = min_max_normalise(sparse_scores)
    else:
        dense, sparse = list(dense_scores), list(sparse_scores)

    sparse_weight = 1.0 - dense_weight
    return [dense_weight * d + sparse_weight * s for d, s in zip(dense, sparse)]
