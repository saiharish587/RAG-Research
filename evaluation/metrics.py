"""Pure metric functions for the SLM-RAG benchmark.

This module deliberately depends on nothing heavier than the standard library
so that every metric is unit-testable on a machine with no GPU, no model
weights and no network access.

Design notes
------------
Three defects in the v1 evaluator are corrected here:

1. **Generation failures were scored as zero accuracy.** An empty response and
   a confidently wrong response both received ``answer_accuracy = 0.0``,
   which silently merged a *generation failure rate* into a *quality metric*.
   Failure is now classified explicitly (:func:`classify_response`) and quality
   metrics return ``None`` for failed generations so they are excluded from
   means rather than dragging them down.

2. **Hallucination was defined as ``1 - accuracy``.** That is an algebraic
   restatement of accuracy, not an independent measurement, and it never looked
   at whether a claim was supported by the retrieved context. It is replaced by
   :func:`groundedness`, which measures the response against the context that
   was actually retrieved.

3. **``precision_at_k`` was not precision@k.** It was bag-of-words overlap
   between the concatenated context and the ground truth. The honest names are
   :func:`context_precision` and :func:`context_recall`; a genuine rank-aware
   :func:`retrieval_hit_rate` is provided separately.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from enum import Enum
from typing import Iterable, Sequence

__all__ = [
    "ResponseStatus",
    "normalise_text",
    "tokenise",
    "content_tokens",
    "classify_response",
    "is_refusal",
    "token_f1",
    "exact_match",
    "groundedness",
    "context_precision",
    "context_recall",
    "retrieval_hit_rate",
    "cosine_similarity",
    "mean_ci",
]

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})

# A compact stop-word list. Kept inline rather than pulled from scikit-learn so
# that this module has no third-party dependency and the metric is stable
# across library versions (a moving stop-word list would silently change
# published numbers between runs).
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself me more most my myself no nor
not of off on once only or other ought our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why with would you your yours yourself yourselves
""".split())

_PUNCT_EXTRA = frozenset(string.punctuation)


class _PunctDropper(dict):
    """Translation table that deletes any punctuation character.

    ``string.punctuation`` covers ASCII only, which is not sufficient here: the
    corpus is OCR'd lecture slides containing typographic quotes, en/em dashes
    and ellipses. Those characters would otherwise survive normalisation and
    make textually identical strings compare unequal.

    Implemented as a ``dict`` subclass so ``str.translate`` memoises the
    category lookup per code point instead of recomputing it per character.
    """

    def __missing__(self, codepoint: int):
        char = chr(codepoint)
        drop = unicodedata.category(char).startswith("P") or char in _PUNCT_EXTRA
        value = None if drop else char
        self[codepoint] = value
        return value


_PUNCT_TABLE = _PunctDropper()
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lower-case, strip punctuation and articles, and collapse whitespace.

    Follows the SQuAD normalisation convention so that ``"The Bell-LaPadula
    model."`` and ``"bell lapadula model"`` compare equal. Unicode is folded to
    NFKC first, because the source corpus is OCR'd lecture slides that contain
    typographic quotes and ligatures.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return _WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()


def tokenise(text: str) -> list[str]:
    """Normalise then split into tokens."""
    normalised = normalise_text(text)
    return normalised.split() if normalised else []


def content_tokens(text: str) -> list[str]:
    """Tokens with stop-words removed.

    Used for groundedness so that the metric is not inflated by function words
    ("the", "of", "is") that appear in any two English texts regardless of
    whether the response actually drew on the retrieved context.
    """
    return [t for t in tokenise(text) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Generation failure classification
# ---------------------------------------------------------------------------


class ResponseStatus(str, Enum):
    """Outcome of a single generation call.

    Only ``OK`` responses are eligible for quality scoring. Everything else is
    a *generation failure* and is reported as a rate, with a denominator,
    alongside quality rather than averaged into it.
    """

    OK = "ok"
    EMPTY = "empty"              # stripped to nothing (the v1 whitespace loop)
    DEGENERATE = "degenerate"    # looped on a tiny vocabulary
    TRUNCATED = "truncated"      # hit the decode ceiling mid-answer
    ERROR = "error"              # transport/runtime failure

    @property
    def is_failure(self) -> bool:
        return self is not ResponseStatus.OK


# A degenerate loop is a long output built from almost no distinct tokens, e.g.
# "the the the ..." or a repeated newline. Thresholds are intentionally
# conservative so that a short legitimate answer is never misfiled: the check
# only applies once the response is long enough for repetition to be
# unambiguous.
_DEGENERATE_MIN_TOKENS = 60
_DEGENERATE_MAX_DISTINCT_RATIO = 0.06


def classify_response(
    response: str | None,
    *,
    generation_tokens: int | None = None,
    token_limit: int | None = None,
    error: bool = False,
) -> ResponseStatus:
    """Classify a raw model response before any quality metric is computed.

    Parameters
    ----------
    response:
        Raw text as returned by the runtime, before stripping.
    generation_tokens, token_limit:
        If the model emitted at least ``token_limit`` tokens the answer was cut
        off, which is a different failure from producing nothing.
    error:
        Set when the runtime raised. Recorded as ``ERROR`` rather than being
        written into the results table as a zero.

    Notes
    -----
    ``EMPTY`` is checked before ``TRUNCATED`` on purpose. In the v1 data 870
    responses generated thousands of tokens that stripped to nothing -- the
    model looped on whitespace until it hit the ceiling. The salient fact is
    that no answer was produced, so ``EMPTY`` takes precedence.
    """
    if error:
        return ResponseStatus.ERROR

    if response is None:
        return ResponseStatus.EMPTY

    stripped = response.strip()
    if not stripped:
        return ResponseStatus.EMPTY

    tokens = stripped.split()
    if len(tokens) >= _DEGENERATE_MIN_TOKENS:
        distinct_ratio = len(set(tokens)) / len(tokens)
        if distinct_ratio <= _DEGENERATE_MAX_DISTINCT_RATIO:
            return ResponseStatus.DEGENERATE

    if token_limit is not None and generation_tokens is not None:
        if generation_tokens >= token_limit:
            return ResponseStatus.TRUNCATED

    return ResponseStatus.OK


# Phrases that indicate the model declined to answer. Tracked separately
# because an abstention is neither a correct answer nor a hallucination, yet
# embedding cosine scores it around 0.83 -- high enough to be mistaken for a
# good answer in the v1 results.
#
# IMPORTANT: these are matched against :func:`normalise_text` output, which has
# already stripped punctuation. Contractions therefore arrive with the
# apostrophe removed ("don't" -> "dont", "can't" -> "cant"), so no pattern here
# may contain an apostrophe -- such a pattern can never fire.
_REFUSAL_PATTERNS = tuple(
    re.compile(p) for p in (
        r"\bdo(?:es)?\s*not\s+(?:contain|provide|specify|mention|include)\b",
        r"\bdoes\s*nt\s+(?:contain|provide|specify|mention|include)\b",
        r"\bnot\s+enough\s+information\b",
        r"\binsufficient\s+(?:information|context|detail)\b",
        r"\b(?:cannot|can\s*not|cant)\s+(?:be\s+)?(?:determine|determined|answer|answered|find|found|tell)\b",
        r"\bi\s+(?:do\s*not|dont|donot)\s+know\b",
        r"\bno\s+(?:relevant\s+|specific\s+|such\s+)?information\b",
        r"\bunable\s+to\s+(?:determine|answer|find)\b",
        r"\bnot\s+(?:specified|mentioned|stated|provided)\s+in\s+(?:context|provided\s+context|text)\b",
    )
)


def is_refusal(response: str | None) -> bool:
    """True when the response is an explicit abstention.

    Abstentions are counted separately from both correct answers and
    hallucinations. A model that reliably says "the context does not contain
    this" is behaving well but scores near zero on :func:`token_f1`; conflating
    that with a wrong answer misrepresents the model, and conflating it with a
    right answer (as embedding cosine effectively did at ~0.83) misrepresents
    the pipeline.
    """
    if not response:
        return False
    return any(p.search(normalise_text(response)) for p in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------


def token_f1(response: str, ground_truth: str) -> float:
    """Token-level F1 between response and ground truth.

    This is the primary quality metric. Unlike embedding cosine -- which spanned
    only 0.746-0.950 across every non-empty v1 response and scored refusals at
    0.83 -- F1 uses the full [0, 1] range and cannot be satisfied by merely
    being on-topic.

    Multiplicity is respected via multiset intersection, so a response cannot
    inflate recall by repeating one correct term.
    """
    pred = tokenise(response)
    gold = tokenise(ground_truth)
    if not pred or not gold:
        return 0.0

    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(response: str, ground_truth: str) -> float:
    """1.0 when normalised strings are identical, else 0.0.

    Reported for completeness. On this benchmark the ground truths are
    multi-sentence explanations, so exact match is expected to be ~0 and should
    not be treated as a headline number; :func:`token_f1` is the discriminative
    counterpart.
    """
    return float(normalise_text(response) == normalise_text(ground_truth))


def groundedness(response: str, retrieved_context: Sequence[str] | None) -> float | None:
    """Fraction of the response's content tokens that appear in the context.

    This is the real hallucination signal that v1 lacked: it asks whether what
    the model said is *supported by the passages it was given*, which is what
    "hallucination" means in the RAG literature.

    Returns ``None`` when there is no retrieved context (the no-RAG arm).
    Groundedness is undefined without context -- returning 0.0 would falsely
    assert that every no-RAG answer is entirely hallucinated, which is exactly
    the class of error that produced the v1 headline result.
    """
    if not retrieved_context:
        return None

    response_terms = content_tokens(response)
    if not response_terms:
        return None

    context_terms = set()
    for chunk in retrieved_context:
        context_terms.update(content_tokens(chunk))
    if not context_terms:
        return None

    supported = sum(1 for t in response_terms if t in context_terms)
    return supported / len(response_terms)


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------


def context_precision(retrieved_context: Sequence[str] | None, ground_truth: str) -> float | None:
    """Share of retrieved content tokens that also occur in the ground truth.

    Renamed from v1's ``precision_at_k``, which measured this quantity but was
    labelled as a rank-aware metric it never computed. Returns ``None`` when
    nothing was retrieved, so the no-RAG arm is excluded from retrieval means
    instead of contributing a structural zero.
    """
    if not retrieved_context:
        return None
    retrieved_terms = [t for c in retrieved_context for t in content_tokens(c)]
    if not retrieved_terms:
        return None
    gold = set(content_tokens(ground_truth))
    if not gold:
        return None
    return sum(1 for t in retrieved_terms if t in gold) / len(retrieved_terms)


def context_recall(retrieved_context: Sequence[str] | None, ground_truth: str) -> float | None:
    """Share of ground-truth content tokens covered by the retrieved context."""
    if not retrieved_context:
        return None
    gold = set(content_tokens(ground_truth))
    if not gold:
        return None
    retrieved_terms = set()
    for chunk in retrieved_context:
        retrieved_terms.update(content_tokens(chunk))
    if not retrieved_terms:
        return None
    return sum(1 for t in gold if t in retrieved_terms) / len(gold)


def retrieval_hit_rate(
    retrieved_context: Sequence[str] | None,
    ground_truth: str,
    *,
    threshold: float = 0.15,
) -> float | None:
    """1.0 if any single retrieved chunk covers ``threshold`` of the gold terms.

    A rank-aware complement to the token-overlap metrics: it answers "did
    retrieval surface at least one genuinely relevant passage?", which the
    diffuse v1 overlap scores (0.03-0.04) could not distinguish from noise.
    """
    if not retrieved_context:
        return None
    gold = set(content_tokens(ground_truth))
    if not gold:
        return None
    for chunk in retrieved_context:
        chunk_terms = set(content_tokens(chunk))
        if chunk_terms and len(gold & chunk_terms) / len(gold) >= threshold:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, guarding zero norms."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def mean_ci(values: Iterable[float], confidence: float = 0.95) -> dict[str, float | int | None]:
    """Mean, standard deviation and a normal-approximation confidence interval.

    Every aggregate the benchmark reports goes through this function so that no
    figure or table can present a bare mean again. The v1 results reported
    means with coefficients of variation up to 535% and no interval at all,
    which is how a difference with p = 0.33 came to be described as a finding.
    """
    data = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(data)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci_low": None, "ci_high": None}

    mean = sum(data) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": 0.0, "ci_low": mean, "ci_high": mean}

    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    sd = math.sqrt(variance)
    # 1.96 for 95%; the normal approximation is adequate at the run counts used
    # here and avoids a SciPy dependency in the metric layer.
    z = 1.96 if abs(confidence - 0.95) < 1e-9 else _z_for(confidence)
    half_width = z * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def _z_for(confidence: float) -> float:
    """Inverse normal CDF via bisection on ``math.erf``."""
    target = (1.0 + confidence) / 2.0
    low, high = 0.0, 10.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0
