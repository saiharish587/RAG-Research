"""Evaluation layer for the SLM-RAG benchmark (v2).

Scores one generation against its ground truth and the context that was
actually retrieved. The pure metric math lives in :mod:`evaluation.metrics`,
which has no third-party dependencies; this module only adds the embedding
model used for the secondary cosine metric.

What a result row now carries
-----------------------------
``status``
    :class:`~evaluation.metrics.ResponseStatus`. Anything other than ``ok`` is
    a *generation failure*, reported as a rate with a denominator and never
    averaged into a quality mean.
``token_f1`` / ``exact_match``
    Lexical quality against ground truth. F1 is the primary quality metric.
``cosine``
    Embedding similarity, retained as a **secondary** metric. In v1 this was
    ``answer_accuracy`` and was the only quality signal; its dynamic range is
    too narrow to discriminate on this benchmark (0.746-0.950 across all
    non-empty responses, with refusals scoring ~0.83).
``groundedness``
    Fraction of the response's content tokens supported by the retrieved
    context -- the actual hallucination signal. v1 defined hallucination as
    ``1 - accuracy``, an algebraic mirror of accuracy that never inspected the
    context. ``None`` on the no-RAG arm, where groundedness is undefined.
``context_precision`` / ``context_recall`` / ``retrieval_hit_rate``
    Retrieval quality. ``None`` when nothing was retrieved, so the no-RAG arm
    is excluded from retrieval means rather than contributing structural zeros.
``is_refusal``
    Explicit abstention, tracked as its own category: neither a correct answer
    nor a hallucination.

Failed generations return ``None`` for every quality metric. The v1 evaluator
wrote ``0.0``, which is how 870 blank generations became a reported "6.97%
accuracy" and manufactured the "RAG Loop Stabilization Effect".
"""

from __future__ import annotations

from typing import Any, Protocol

from evaluation.metrics import (
    ResponseStatus,
    classify_response,
    context_precision,
    context_recall,
    cosine_similarity,
    exact_match,
    groundedness,
    is_refusal,
    retrieval_hit_rate,
    token_f1,
)

# Quality metrics are undefined for a failed generation. Named once so the
# failure path cannot drift out of sync with the success path.
_QUALITY_FIELDS = (
    "token_f1",
    "exact_match",
    "cosine",
    "groundedness",
    "context_precision",
    "context_recall",
    "retrieval_hit_rate",
)


class EmbeddingModel(Protocol):
    """Anything exposing sentence-transformers' ``encode`` interface."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:  # -> array (n, dim)
        ...


class Evaluator:
    """Scores a single generation.

    Parameters
    ----------
    embedding_model_name, device:
        Used to lazily construct a :class:`SentenceTransformer` on first use.
    embedding_model:
        A pre-built encoder. When supplied it is used as-is and
        sentence-transformers is never imported, which is what makes this class
        testable on a machine with no model weights, no GPU and no network.
    token_limit:
        Decode ceiling for the run, used to detect truncated answers.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        embedding_model: EmbeddingModel | None = None,
        token_limit: int | None = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.token_limit = token_limit
        self._model = embedding_model

    # ------------------------------------------------------------------
    # Embedding model (lazy)
    # ------------------------------------------------------------------

    @property
    def model(self) -> EmbeddingModel:
        """The encoder, constructed on first access.

        Imported inside the property rather than at module scope so that
        importing this module does not require sentence-transformers.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._model

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def evaluate(
        self,
        response: str | None,
        ground_truth: str,
        retrieved_context: list[str] | None,
        *,
        generation_tokens: int | None = None,
        prompt_tokens: int | None = None,
        latency: float | None = None,
        error: bool = False,
    ) -> dict[str, Any]:
        """Score one generation.

        ``retrieved_context=None`` (or empty) denotes the no-RAG arm, where
        every context-dependent metric is ``None`` rather than zero.
        """
        status = classify_response(
            response,
            generation_tokens=generation_tokens,
            token_limit=self.token_limit,
            error=error,
        )

        row: dict[str, Any] = {
            "status": status.value,
            "is_failure": status.is_failure,
            "is_refusal": bool(response) and not error and is_refusal(response),
            # Cost counters are recorded for every run, failures included, so
            # the latency distribution stays honest.
            "generation_tokens": generation_tokens,
            "prompt_tokens": prompt_tokens,
            "latency": latency,
        }

        if status.is_failure:
            row.update(dict.fromkeys(_QUALITY_FIELDS))
            return row

        # status == OK guarantees non-empty text.
        text = response or ""
        row.update(
            token_f1=token_f1(text, ground_truth),
            exact_match=exact_match(text, ground_truth),
            cosine=self._cosine(text, ground_truth),
            groundedness=groundedness(text, retrieved_context),
            context_precision=context_precision(retrieved_context, ground_truth),
            context_recall=context_recall(retrieved_context, ground_truth),
            retrieval_hit_rate=retrieval_hit_rate(retrieved_context, ground_truth),
        )
        return row

    def evaluate_run(
        self,
        query: str,
        response: str | None,
        retrieved_context: list[str] | None,
        ground_truth: str,
        stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Orchestration entry point: score a run and attach its raw fields.

        Kept signature-compatible with the v1 evaluator so the benchmark driver
        needs no change, but the returned metrics are the v2 set.
        """
        stats = stats or {}
        row = self.evaluate(
            response,
            ground_truth,
            retrieved_context,
            generation_tokens=stats.get("generation_tokens"),
            prompt_tokens=stats.get("prompt_tokens"),
            latency=stats.get("latency"),
            error=bool(stats.get("error")),
        )
        return {
            "query": query,
            "response": response,
            "ground_truth": ground_truth,
            "n_retrieved": len(retrieved_context) if retrieved_context else 0,
            "generation_speed": stats.get("generation_speed"),
            "prompt_eval_speed": stats.get("prompt_eval_speed"),
            **row,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cosine(self, response: str, ground_truth: str) -> float | None:
        """Embedding cosine between response and ground truth.

        A failure to encode returns ``None``, not ``0.0``: an infrastructure
        problem must not masquerade as a maximally wrong answer.
        """
        try:
            vectors = self.model.encode([response, ground_truth])
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[evaluator] embedding failed, cosine recorded as None: {exc}")
            return None
        return cosine_similarity(vectors[0], vectors[1])
