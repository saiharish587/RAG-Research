"""Advanced RAG: query rewriting, hybrid retrieval and cross-encoder reranking.

Three v1 defects are corrected here.

**Hybrid fusion mixed incomparable score scales.** ``0.7 * dense + 0.3 * sparse``
combined FAISS cosine (~0.5-0.9 on this corpus) with TF-IDF cosine (~0.0-0.2).
The nominal 70/30 blend therefore behaved close to pure dense retrieval, and
the sparse arm of the "hybrid" search contributed almost nothing. Both signals
are now min-max normalised before weighting (:func:`rag.fusion.weighted_score_fusion`).

**The TF-IDF index was refit on every query.** ``fit_transform(corpus)`` ran
inside the search path, so the entire corpus was re-vectorised once per query.
This is pure waste, and because it sat inside the untimed portion of the v1
pipeline it never appeared in the reported latency. The vectoriser is now fit
once and cached.

**Dense results were matched to corpus chunks by nested text comparison**, an
O(n^2) string scan per query. Replaced with a text-to-index map built with the
cached vectoriser.

``rerank_top_n`` is now explicit rather than implied by ``top_k * 2 * 2``, so
the number of candidates entering the cross-encoder is a stated parameter.
"""

from __future__ import annotations

import time
from typing import Any

from rag.base import RunRecorder
from rag.fusion import min_max_normalise, weighted_score_fusion
from rag.prompts import build_answer_prompt, build_rewrite_prompt
from rag.sparse import TfidfIndex, corpus_signature

# A leading "label:" is stripped from a rewritten query only when the label is
# short enough to be a label rather than prose. 24 characters covers
# "Optimized Query" (15) and "Optimized Search Query" (22).
_MAX_LABEL_LEN = 24

# ...and only when it reads like a label. Without this, a colon inside a
# legitimate query ("Docker: an introduction") would truncate the query.
_LABEL_WORDS = (
    "query", "question", "search", "rewrite", "rewritten",
    "optimized", "optimised", "output", "version", "terms",
)

# Shorter than this is not a usable search query, so fall back to the original.
_MIN_QUERY_LEN = 3


class AdvancedRAGPipeline:
    """Rewrite the query, retrieve with dense+sparse fusion, then rerank."""

    name = "advanced"

    def __init__(
        self,
        db_manager,
        generator,
        top_k: int = 3,
        rerank: bool = True,
        rerank_model_name: str = "BAAI/bge-reranker-base",
        reranker=None,
        rerank_top_n: int = 12,
        dense_weight: float = 0.7,
        sparse_index_factory=TfidfIndex,
        clock=time.perf_counter,
    ):
        self.db_manager = db_manager
        self.generator = generator
        self.top_k = top_k
        self.rerank = rerank
        self.clock = clock
        # Number of hybrid candidates scored by the cross-encoder. Stated
        # explicitly: in v1 this was an emergent product of two separate
        # doublings and was never reported.
        self.rerank_top_n = rerank_top_n
        self.dense_weight = dense_weight

        # Lazily built sparse index, cached against a signature of the chunk
        # list it was fit on so a rebuilt corpus invalidates the cache instead
        # of silently scoring against stale vocabulary. The factory is
        # injectable so the caching and fusion logic is testable without
        # scikit-learn installed.
        self._sparse_index_factory = sparse_index_factory
        self._sparse_index = None
        self._sparse_signature = None
        self._text_to_index: dict[str, int] = {}
        self._sparse_fit_count = 0

        # The reranker may be injected, which keeps this class constructible in
        # tests without downloading cross-encoder weights.
        self.reranker = reranker
        if self.rerank and self.reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                print(f"Loading reranker model: {rerank_model_name}...")
                self.reranker = CrossEncoder(rerank_model_name)
            except Exception as exc:
                print(f"Error loading reranker {rerank_model_name}: {exc}. Disabling reranking.")
                self.rerank = False

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    def rewrite_query(self, query: str, recorder=None) -> str:
        """Rephrase the query for retrieval, falling back to the original.

        The fallback matters for interpretation: when rewriting fails the arm
        degrades to searching the original query, not to searching nothing.
        """
        prompt = build_rewrite_prompt(query)
        if recorder is None:
            response = self.generator.generate(prompt)
        else:
            response = recorder.generate(self.generator, prompt, stage="rewrite")
        rewritten = (response.get("response") or "").strip()

        # Reject a failed generation *before* any cleanup. v1 stripped the
        # leading "ERROR:" label first, after which the "ERROR" test could no
        # longer match and the error text was used as a search query.
        if response.get("error") or not rewritten or "ERROR" in rewritten.upper():
            return query

        # Strip a leading label such as "Optimized Query:". v1 used
        # `len(prefix) < 15`, but "Optimized Query" is exactly 15 characters --
        # the one label the prompt actually asks for was never stripped.
        head, separator, tail = rewritten.partition(":")
        if (
            separator
            and len(head) <= _MAX_LABEL_LEN
            and "\n" not in head
            and any(word in head.lower() for word in _LABEL_WORDS)
        ):
            candidate = tail.strip()
            if candidate:
                rewritten = candidate

        rewritten = rewritten.strip().strip('"').strip("'").strip()
        if len(rewritten) < _MIN_QUERY_LEN:
            return query
        return rewritten

    # ------------------------------------------------------------------
    # Sparse index (fit once)
    # ------------------------------------------------------------------

    def _ensure_sparse_index(self, chunks) -> bool:
        """Build the sparse index once per chunk set, reusing it thereafter."""
        texts = [c["text"] for c in chunks]
        signature = corpus_signature(texts)
        if self._sparse_index is not None and self._sparse_signature == signature:
            return True
        try:
            self._sparse_index = self._sparse_index_factory(texts)
        except Exception as exc:
            # Retrieval degrades to dense-only rather than failing the run, but
            # it is reported so a silently dense-only "hybrid" arm is visible.
            print(f"Sparse index unavailable, falling back to dense-only: {exc}")
            self._sparse_index = None
            self._sparse_signature = None
            return False
        self._sparse_signature = signature
        self._text_to_index = {text: i for i, text in enumerate(texts)}
        self._sparse_fit_count += 1
        return True

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Dense + sparse retrieval fused on a common [0, 1] scale.

        Each arm is scaled on the range over which it was actually observed,
        which is not the same range for both. Sparse scores exist for every
        chunk in the corpus, so the sparse arm is scaled globally. Dense scores
        exist only for what the vector store returned, so the dense arm is
        scaled over that retrieved set.

        A chunk found by one arm but not the other is placed at the bottom of
        the missing arm's *normalised* scale. It must not be given a raw score
        of 0.0 before normalisation: a sentinel below every genuine score
        becomes the minimum, and min-max then compresses all the real scores
        into the top sliver of the range. On this corpus that turned a 0.03
        spread of dense similarities into a spread of 0.03/0.90, making four
        distinguishable chunks effectively tied.
        """
        n_candidates = max(top_k, self.rerank_top_n)
        dense_results = self.db_manager.search(query, top_k=n_candidates)

        chunks = self.db_manager.doc_chunks
        if not chunks or not self._ensure_sparse_index(chunks):
            return dense_results

        sparse_scores = self._sparse_index.scores(query)
        sparse_scaled = min_max_normalise(sparse_scores)

        # Dense arm: scaled over the retrieved set, which is the only range for
        # which scores are observable.
        dense_indices, dense_raw = [], []
        for result in dense_results:
            idx = self._text_to_index.get(result["chunk"]["text"])
            if idx is not None:
                dense_indices.append(idx)
                dense_raw.append(result["score"])
        dense_scaled = dict(zip(dense_indices, min_max_normalise(dense_raw)))

        # Union of both candidate sets: a chunk strong on either signal must be
        # eligible. v1 fused over the whole corpus, so every chunk the dense
        # arm had not returned still entered the blend with a dense score of
        # 0.0 -- and unnormalised, that was indistinguishable from a genuine
        # low similarity.
        candidates = list(dense_indices)
        seen = set(dense_indices)
        sparse_ranked = sorted(
            range(len(chunks)), key=lambda i: sparse_scores[i], reverse=True
        )[:n_candidates]
        for idx in sparse_ranked:
            if sparse_scores[idx] > 0 and idx not in seen:
                seen.add(idx)
                candidates.append(idx)

        if not candidates:
            return dense_results

        fused = weighted_score_fusion(
            # Absent from the dense results -> bottom of the dense scale.
            [dense_scaled.get(i, 0.0) for i in candidates],
            # Sparse scores exist corpus-wide, so this is always a real value.
            [sparse_scaled[i] for i in candidates],
            dense_weight=self.dense_weight,
            normalise=False,  # both arms are already on their own [0, 1] scale
        )

        results = [
            {"chunk": chunks[idx], "score": score}
            for idx, score in zip(candidates, fused)
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:n_candidates]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict[str, Any]:
        recorder = RunRecorder(clock=self.clock)

        # Every stage below is timed. In v1 only the final generate() call was,
        # so this arm's two extra costs -- the rewrite call and the
        # cross-encoder pass -- were invisible in the reported latency.
        optimized_query = self.rewrite_query(query, recorder=recorder)

        with recorder.stage("retrieve"):
            candidates = self.hybrid_search(optimized_query, top_k=self.rerank_top_n)

        if self.rerank and self.reranker and candidates:
            with recorder.stage("rerank"):
                pairs = [[optimized_query, c["chunk"]["text"]] for c in candidates]
                scores = self.reranker.predict(pairs)
                for score, candidate in zip(scores, candidates):
                    candidate["rerank_score"] = float(score)
                candidates = sorted(candidates, key=lambda c: c["rerank_score"],
                                    reverse=True)

        retrieved_contexts = [c["chunk"]["text"] for c in candidates[: self.top_k]]

        prompt = build_answer_prompt(query, retrieved_contexts)
        final = recorder.generate(self.generator, prompt, stage="generate")
        return recorder.finish(
            final,
            retrieved_contexts,
            rewritten_query=optimized_query,
            query_was_rewritten=optimized_query != query,
        )
